"""PoSafeNet models for the 2D robot (unicycle) navigation example.

Each head enforces the three obstacle HOCBFs by sequential closed-form
half-space projection in a fixed, head-specific total order (one permutation of
the obstacles per head; see `dCBF_totalorder_proj`). The heads are combined by
learned weights `wt`:

PoSafeNet_Hard    : Gumbel-softmax head selection in training, argmax at inference.
PoSafeNet_Mixture : softmax-weighted mixture over heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class PoSafeNet_Hard(nn.Module):
    """
    ABNet-style multi-head network, but safety filter is sequential halfspace projection
    over 3 disk obstacles, using a fixed total order per head.
    """

    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn, heads=10):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.device = device

        self.mean = torch.from_numpy(mean).to(device)
        self.std  = torch.from_numpy(std).to(device)

        # obstacles (K,3) = (ox,oy,R)
        self.obs = torch.tensor(
            [[35.0, 0.0, 9.0],
             [20.0, 16.0, 8.0],
             [49.0, 10.0, 7.0]],
            dtype=torch.float64,
            device=device,
        )
        self.obs_x = float(self.obs[0, 0].item())
        self.obs_y = float(self.obs[0, 1].item())
        self.R     = float(self.obs[0, 2].item())

        self.heads = heads
        self.add_noise = False

        # all permutations for 3 disks
        self.orders = [
            ("obs0", "obs1", "obs2"),
            ("obs0", "obs2", "obs1"),
            ("obs1", "obs0", "obs2"),
            ("obs1", "obs2", "obs0"),
            ("obs2", "obs1", "obs0"),
            ("obs2", "obs0", "obs1"),
        ]

        # BN layers
        if bn:
            self.bn1  = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        # multi-head MLPs (same as ABNet)
        self.fc1, self.fc21, self.fc22, self.fc31, self.fc32 = (
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        )
        for _ in range(self.heads):
            self.fc1.append(nn.Linear(nFeatures, nHidden1).double().to(device))
            self.fc21.append(nn.Linear(nHidden1, nHidden21).double().to(device))
            self.fc22.append(nn.Linear(nHidden1, nHidden22).double().to(device))
            self.fc31.append(nn.Linear(nHidden21, nCls).double().to(device))  # u_nom (physical)
            self.fc32.append(nn.Linear(nHidden22, nCls).double().to(device))  # kappa

        # mixture weights (same as ABNet)
        self.wt = Parameter(torch.ones(self.heads, dtype=torch.float64, device=device))

        # keep your "memory" placeholder if you want
        self.x32 = 0

    # ============================================================
    # Halfspace projection
    # ============================================================
    @staticmethod
    def proj_halfspace_batch(u: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
        # project onto { u | a^T u >= b }
        if b.ndim == 2 and b.shape[1] == 1:
            b = b[:, 0]
        a = a.to(dtype=u.dtype, device=u.device)
        b = b.to(dtype=u.dtype, device=u.device)

        au = (a * u).sum(dim=1)          # (B,)
        viol = b - au                    # (B,)
        denom = (a * a).sum(dim=1) + eps # (B,)
        step = torch.relu(viol) / denom  # (B,)
        return u + step[:, None] * a

    # ============================================================
    # One-disk HOCBF projection (your function)
    # ============================================================
    def _proj_disk_hocbf(self, u, px, py, theta, v, k1, k2, return_ab: bool = False):
        device = u.device
        dtype  = u.dtype

        ox = torch.as_tensor(self.obs_x, device=device, dtype=dtype)
        oy = torch.as_tensor(self.obs_y, device=device, dtype=dtype)
        R  = torch.as_tensor(self.R,     device=device, dtype=dtype)

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        barrier = (px - ox)**2 + (py - oy)**2 - R**2
        barrier_dot = 2*(px - ox)*v*cos_t + 2*(py - oy)*v*sin_t

        Lf2b = 2*(v**2)

        Lg1 = -2*(px - ox)*v*sin_t + 2*(py - oy)*v*cos_t
        Lg2 =  2*(px - ox)*cos_t    + 2*(py - oy)*sin_t

        h = Lf2b + (k1 + k2)*barrier_dot + (k1*k2)*barrier

        a = torch.stack([Lg1, Lg2], dim=1)    # (B,2)
        b = (-h).reshape(-1)                  # (B,)

        u_proj = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u_proj, a, b, h.reshape(-1)
        return u_proj

    # ============================================================
    # Sequential projection over 3 disks following head-specific order
    # ============================================================
    def dCBF_totalorder_proj(self, x0, u_nom, kappa, sgn, nBatch, i):
        # state: [px, py, theta, v]
        px    = x0[:, 0]
        py    = x0[:, 1]
        theta = x0[:, 2]
        v     = x0[:, 3]

        u = u_nom  # (B,2)

        order = self.orders[i % len(self.orders)]

        # kappa: use first two dims as (k1,k2)
        k1 = kappa[:, 0]
        k2 = kappa[:, 1]

        # save & restore obstacle params
        ox0, oy0, R0 = self.obs_x, self.obs_y, self.R
        try:
            for name in order:
                j = int(name[3:])  # "obs0" -> 0
                self.obs_x = float(self.obs[j, 0].item())
                self.obs_y = float(self.obs[j, 1].item())
                self.R     = float(self.obs[j, 2].item())

                u = self._proj_disk_hocbf(u, px, py, theta, v, k1, k2, return_ab=False)
        finally:
            self.obs_x, self.obs_y, self.R = ox0, oy0, R0

        return u

    # ============================================================
    # One head forward
    # ============================================================
    def sub_forward(self, x, x0, nBatch, sgn, i):
        if self.add_noise:
            x = F.relu(self.fc1[i](x + 0.1 * torch.randn_like(x)))
        else:
            x = F.relu(self.fc1[i](x))
        if self.bn:
            x = self.bn1(x)

        x21 = F.relu(self.fc21[i](x))
        if self.bn:
            x21 = self.bn21(x21)

        x22 = F.relu(self.fc22[i](x))
        if self.bn:
            x22 = self.bn22(x22)

        u_nom = self.fc31[i](x21)                 # (B,2) physical
        kappa = 4.0 * torch.sigmoid(self.fc32[i](x22))  # (B,2+) positive

        if i == 0:
            self.x32 = kappa

        u_safe = self.dCBF_totalorder_proj(x0, u_nom, kappa, sgn, nBatch, i)
        return u_safe
    
    def forward(self, x, sgn, itr=0, tau=1.0, hard=True):
        nBatch = x.size(0)
        x = x.view(nBatch, -1).to(self.mean.dtype)
        x0 = x * self.std + self.mean

        if not self.training:
            # deterministic inference: pick the best head globally
            idx = torch.argmax(self.wt).item()
            return self.sub_forward(x, x0, nBatch, sgn, idx)

        # training: gumbel selection
        y = F.gumbel_softmax(self.wt, tau=tau, hard=hard)  # (heads,)
        u_sel = 0.0
        for i in range(self.heads):
            u_sel = u_sel + y[i] * self.sub_forward(x, x0, nBatch, sgn, i)
        return u_sel



class PoSafeNet_Mixture(nn.Module):
    """
    ABNet-style multi-head network, but safety filter is sequential halfspace projection
    over 3 disk obstacles, using a fixed total order per head.
    """

    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn, heads=10):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.device = device

        self.mean = torch.from_numpy(mean).to(device)
        self.std  = torch.from_numpy(std).to(device)

        # obstacles (K,3) = (ox,oy,R)
        self.obs = torch.tensor(
            [[35.0, 0.0, 9.0],
             [20.0, 16.0, 8.0],
             [49.0, 10.0, 7.0]],
            dtype=torch.float64,
            device=device,
        )
        self.obs_x = float(self.obs[0, 0].item())
        self.obs_y = float(self.obs[0, 1].item())
        self.R     = float(self.obs[0, 2].item())

        self.heads = heads
        self.add_noise = False

        # all permutations for 3 disks
        self.orders = [
            ("obs0", "obs1", "obs2"),
            ("obs0", "obs2", "obs1"),
            ("obs1", "obs0", "obs2"),
            ("obs1", "obs2", "obs0"),
            ("obs2", "obs1", "obs0"),
            ("obs2", "obs0", "obs1"),
        ]
        

        # BN layers
        if bn:
            self.bn1  = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        # multi-head MLPs (same as ABNet)
        self.fc1, self.fc21, self.fc22, self.fc31, self.fc32 = (
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        )
        for _ in range(self.heads):
            self.fc1.append(nn.Linear(nFeatures, nHidden1).double().to(device))
            self.fc21.append(nn.Linear(nHidden1, nHidden21).double().to(device))
            self.fc22.append(nn.Linear(nHidden1, nHidden22).double().to(device))
            self.fc31.append(nn.Linear(nHidden21, nCls).double().to(device))  # u_nom (physical)
            self.fc32.append(nn.Linear(nHidden22, nCls).double().to(device))  # kappa

        # mixture weights (same as ABNet)
        self.wt = Parameter(torch.ones(self.heads, dtype=torch.float64, device=device))

        # keep your "memory" placeholder if you want
        self.x32 = 0
        self.kappa_last = [None] * heads

    # ============================================================
    # Halfspace projection
    # ============================================================
    @staticmethod
    def proj_halfspace_batch(u: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
        # project onto { u | a^T u >= b }
        if b.ndim == 2 and b.shape[1] == 1:
            b = b[:, 0]
        a = a.to(dtype=u.dtype, device=u.device)
        b = b.to(dtype=u.dtype, device=u.device)

        au = (a * u).sum(dim=1)          # (B,)
        viol = b - au                    # (B,)
        denom = (a * a).sum(dim=1) + eps # (B,)
        step = torch.relu(viol) / denom  # (B,)
        return u + step[:, None] * a

    # ============================================================
    # One-disk HOCBF projection (your function)
    # ============================================================
    def _proj_disk_hocbf(self, u, px, py, theta, v, k1, k2, return_ab: bool = False):
        device = u.device
        dtype  = u.dtype

        ox = torch.as_tensor(self.obs_x, device=device, dtype=dtype)
        oy = torch.as_tensor(self.obs_y, device=device, dtype=dtype)
        R  = torch.as_tensor(self.R,     device=device, dtype=dtype)

        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)

        barrier = (px - ox)**2 + (py - oy)**2 - R**2
        barrier_dot = 2*(px - ox)*v*cos_t + 2*(py - oy)*v*sin_t

        Lf2b = 2*(v**2)

        Lg1 = -2*(px - ox)*v*sin_t + 2*(py - oy)*v*cos_t
        Lg2 =  2*(px - ox)*cos_t    + 2*(py - oy)*sin_t

        h = Lf2b + (k1 + k2)*barrier_dot + (k1*k2)*barrier

        a = torch.stack([Lg1, Lg2], dim=1)    # (B,2)
        b = (-h).reshape(-1)                  # (B,)

        u_proj = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u_proj, a, b, h.reshape(-1)
        return u_proj

    # ============================================================
    # Sequential projection over 3 disks following head-specific order
    # ============================================================
    def dCBF_totalorder_proj(self, x0, u_nom, kappa, sgn, nBatch, i):
        # state: [px, py, theta, v]
        px    = x0[:, 0]
        py    = x0[:, 1]
        theta = x0[:, 2]
        v     = x0[:, 3]

        u = u_nom  # (B,2)

        order = self.orders[i % len(self.orders)]

        # kappa: use first two dims as (k1,k2)
        k1 = kappa[:, 0]
        k2 = kappa[:, 1]

        # save & restore obstacle params
        ox0, oy0, R0 = self.obs_x, self.obs_y, self.R
        try:
            for name in order:
                j = int(name[3:])  # "obs0" -> 0
                self.obs_x = float(self.obs[j, 0].item())
                self.obs_y = float(self.obs[j, 1].item())
                self.R     = float(self.obs[j, 2].item())

                u = self._proj_disk_hocbf(u, px, py, theta, v, k1, k2, return_ab=False)
        finally:
            self.obs_x, self.obs_y, self.R = ox0, oy0, R0

        return u

    # ============================================================
    # One head forward
    # ============================================================
    def sub_forward(self, x, x0, nBatch, sgn, i):
        if self.add_noise:
            x = F.relu(self.fc1[i](x + 0.1 * torch.randn_like(x)))
        else:
            x = F.relu(self.fc1[i](x))
        if self.bn:
            x = self.bn1(x)

        x21 = F.relu(self.fc21[i](x))
        if self.bn:
            x21 = self.bn21(x21)

        x22 = F.relu(self.fc22[i](x))
        if self.bn:
            x22 = self.bn22(x22)

        u_nom = self.fc31[i](x21)                 # (B,2) physical
        kappa = 4.0 * torch.sigmoid(self.fc32[i](x22))  # (B,2+) positive

        # if i == 0:
        #     self.x32 = kappa

        if i == 0:
            self.x32 = kappa
        self.kappa_last[i] = kappa
        u_safe = self.dCBF_totalorder_proj(x0, u_nom, kappa, sgn, nBatch, i)
        return u_safe

    # ============================================================
    # Mixture over heads
    # ============================================================
    def forward(self, x, sgn, itr=0):
        nBatch = x.size(0)
        x = x.view(nBatch, -1).to(self.mean.dtype)
        x0 = x * self.std + self.mean

        wt_sum = torch.sum(torch.exp(self.wt), dim=0)

        u_sum = 0.0
        for i in range(self.heads):
            u_i = self.sub_forward(x, x0, nBatch, sgn, i)
            u_sum = u_sum + u_i * torch.exp(self.wt[i]) / wt_sum

        return u_sum

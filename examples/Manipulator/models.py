"""PoSafeNet models for the two-link planar manipulator example.

Safety constraints (HOCBFs, relative degree 2) on u = [u1, u2]:
  * tip obstacle disk            (tip_disk_ab)
  * joint-angle difference limits phi_min / phi_max (phi_min_ab, phi_max_ab)

PoSafeNet_Mixture : softmax-weighted mixture over priority heads.
PoSafeNet_Hard    : Gumbel-softmax head selection in training, argmax at inference.

The hs_margin / antiparallel_intersection_stats / summarize_noninterference_*
helpers are used by the models' optional non-interference diagnostics.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

def _wrap_pi_torch(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2.0 * torch.pi) - torch.pi

def phi_min_ab(theta1, w1, theta2, w2, k1=2.0, k2=2.0, phi_min=-2.0):
    phi = _wrap_pi_torch(theta2 - theta1)
    phi_dot = w2 - w1

    barrier = phi - phi_min
    barrier_dot = phi_dot
    Lf2b = torch.zeros_like(phi)

    # phi_ddot = u2 - u1
    Lg1 = -torch.ones_like(phi)
    Lg2 = +torch.ones_like(phi)

    h = Lf2b + (k1 + k2) * barrier_dot + (k1 * k2) * barrier
    a = torch.stack([Lg1, Lg2], dim=1)     # (B,2)
    b = (-h)                               # (B,)
    return a, b

def phi_max_ab(theta1, w1, theta2, w2, k1=2.0, k2=2.0, phi_max=2.0):
    phi = _wrap_pi_torch(theta2 - theta1)
    phi_dot = w2 - w1

    barrier = phi_max - phi
    barrier_dot = -phi_dot
    Lf2b = torch.zeros_like(phi)

    # barrier_ddot = u1 - u2
    Lg1 = +torch.ones_like(phi)
    Lg2 = -torch.ones_like(phi)

    h = Lf2b + (k1 + k2) * barrier_dot + (k1 * k2) * barrier
    a = torch.stack([Lg1, Lg2], dim=1)
    b = (-h)
    return a, b

def tip_disk_ab(theta1, w1, theta2, w2, k1_tip, k2_tip, obs_x, obs_y, R):
    l1, l2 = 3.0, 3.0
    ox, oy = float(obs_x), float(obs_y)
    R = float(R)

    sint1 = torch.sin(theta1); cost1 = torch.cos(theta1)
    sint2 = torch.sin(theta2); cost2 = torch.cos(theta2)

    px = l1 * cost1 + l2 * cost2
    py = l1 * sint1 + l2 * sint2

    barrier = (px - ox) ** 2 + (py - oy) ** 2 - R ** 2
    barrier_dot = 2 * (px - ox) * (-l1 * sint1 * w1 - l2 * sint2 * w2) + \
                  2 * (py - oy) * ( l1 * cost1 * w1 + l2 * cost2 * w2)

    Lf2b = 2 * (-l1 * sint1 * w1 - l2 * sint2 * w2) ** 2 + \
           2 * ( l1 * cost1 * w1 + l2 * cost2 * w2) ** 2 + \
           2 * (px - ox) * (-l1 * cost1 * w1 ** 2 - l2 * cost2 * w2 ** 2) + \
           2 * (py - oy) * (-l1 * sint1 * w1 ** 2 - l2 * sint2 * w2 ** 2)

    Lg1 = 2 * (px - ox) * (-l1 * sint1) + 2 * (py - oy) * (l1 * cost1)
    Lg2 = 2 * (px - ox) * (-l2 * sint2) + 2 * (py - oy) * (l2 * cost2)

    h = Lf2b + (k1_tip + k2_tip) * barrier_dot + (k1_tip * k2_tip) * barrier
    a = torch.stack([Lg1, Lg2], dim=1)   # (B,2)
    b = (-h)                             # (B,)
    return a, b

def hs_margin(
    u: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Margin for halfspace a^T u >= b.
    Positive means satisfied.
    """
    if b.ndim == 2 and b.shape[1] == 1:
        b = b[:, 0]

    return torch.sum(a * u, dim=1) - b


def antiparallel_intersection_stats(
    a1: torch.Tensor,
    b1: torch.Tensor,
    a2: torch.Tensor,
    b2: torch.Tensor,
    eps: float = 1e-15,
):
    """
    Diagnostics for halfspaces a1^T u >= b1 and a2^T u >= b2.

    When the normals are antiparallel, intersection_slack >= 0 is
    equivalent to a nonempty intersection. The cosine is returned so the
    caller can verify antiparallelity rather than assuming it.
    """
    if b1.ndim == 2:
        b1 = b1[:, 0]
    if b2.ndim == 2:
        b2 = b2[:, 0]

    n1 = torch.linalg.vector_norm(a1, dim=1).clamp_min(eps)
    n2 = torch.linalg.vector_norm(a2, dim=1).clamp_min(eps)

    cosine = torch.sum(a1 * a2, dim=1) / (n1 * n2)
    intersection_slack = -(b1 / n1 + b2 / n2)
    return cosine, intersection_slack


def summarize_noninterference_records(
    records,
    tau: float = 1e-7,
    active_tol: float = 1e-12,
):
    """Summarize records produced by PoSafeNet's audit mode."""
    eligible = [
        r for r in records
        if r["preserved_margin_before"] >= -tau
        and r["correction_norm"] > active_tol
    ]
    interference = [
        r for r in eligible
        if r["preserved_margin_after"] < -tau
    ]

    return {
        "tau": float(tau),
        "active_tol": float(active_tol),
        "records": len(records),
        "eligible": len(eligible),
        "interference": len(interference),
        "worst_preserved_margin_after": (
            min(r["preserved_margin_after"] for r in eligible)
            if eligible else float("nan")
        ),
        "max_projection_residual": (
            max(max(0.0, -r["projected_margin_after"]) for r in records)
            if records else float("nan")
        ),
        "empty_intersections": sum(
            r["intersection_slack"] < -tau for r in records
        ),
        "worst_intersection_slack": (
            min(r["intersection_slack"] for r in records)
            if records else float("nan")
        ),
        "max_antiparallel_error": (
            max(abs(r["normal_cosine"] + 1.0) for r in records)
            if records else float("nan")
        ),
    }


def summarize_noninterference_sweep(
    records,
    taus=(1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5),
    active_tol: float = 1e-12,
):
    return [
        summarize_noninterference_records(records, tau=tau, active_tol=active_tol)
        for tau in taus
    ]


class PoSafeNet_Mixture(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn, heads = 4):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.obs_x = 0  #obstacle location
        self.obs_y = 7
        self.R = 4   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.nCls = model_param[7]

        self.x52 = 0
        self.heads = heads

        self.add_noise = False

        # Audit-only state. These are plain Python attributes, so adding them
        # does not change checkpoint compatibility or trained parameters.
        self.audit_enabled = False
        self.last_audit = []
        self.last_head_outputs = {}
        self.last_output_margins = {}
        self.last_mixture_weights = None

        # self.orders = [
        #     ("phi_min", "phi_max", "tip",   "elbow"),
        #     ("phi_min", "phi_max", "elbow", "tip"),
        #     ("phi_max", "phi_min", "tip",   "elbow"),
        #     ("phi_max", "phi_min", "elbow", "tip"),
        # ]

        self.orders = [
            # ("phi_min", "phi_max", "elbow", "tip"),
            # ("phi_min", "phi_max", "elbow", "tip"),
            ("tip","phi_min", "phi_max"),
            # ("phi_max", "phi_min", "elbow", "tip"),
            # ("phi_max", "phi_min", "elbow", "tip"),
            ("tip","phi_max", "phi_min"),
        ]    
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1, self.fc2, self.fc31, self.fc32, self.fc41, self.fc42, self.fc51, self.fc52 = nn.ModuleList(), nn.ModuleList(), \
              nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(self.heads):
            self.fc1.append(nn.Linear(model_param[0], model_param[1]).double().to(device))
            self.fc2.append(nn.Linear(model_param[1], model_param[2]).double().to(device))
            self.fc31.append(nn.Linear(model_param[2], model_param[3]).double().to(device))
            self.fc32.append(nn.Linear(model_param[2], model_param[4]).double().to(device))
            self.fc41.append(nn.Linear(model_param[3], model_param[5]).double().to(device))
            self.fc42.append(nn.Linear(model_param[4], model_param[6]).double().to(device))
            self.fc51.append(nn.Linear(model_param[5], model_param[7]).double().to(device))
            # self.fc52.append(nn.Linear(model_param[6], model_param[7]).double().to(device))
            self.fc52.append(nn.Linear(model_param[6], 2).double().to(device))

        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # QP params.
        # from previous layers
        self.wt = Parameter(torch.ones(self.heads))
    
    def sub_forward(self, x, x0, nBatch, sgn, i): 
        if self.add_noise:    
            x1 = self.activation(self.fc1[i](x + 0.1*torch.randn_like(x)))  #add noise to each model observation
        else:
            x1 = self.activation(self.fc1[i](x))
        if self.bn:
            x1 = self.bn1(x1)
        
        x2 = self.activation(self.fc2[i](x1))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = self.activation(self.fc31[i](x2))
        x32 = self.activation(self.fc32[i](x2))

        x41 = self.activation(self.fc41[i](x31))
        x42 = self.activation(self.fc42[i](x32))

        x51 = self.fc51[i](x41)
        x52 = 4*nn.Sigmoid()(self.fc52[i](x42))  # ensure CBF parameters are positive

        if i == 0:
            self.x52 = x52
        
        # BarrierNet/ABNet
        u = self.dCBF_totalorder_proj(x0, x51, x52, sgn, nBatch, i)
        return u

    def forward(self, x, sgn, itr=0):
        if self.audit_enabled:
            self.last_audit = []
            self.last_head_outputs = {}
            self.last_output_margins = {}
            self.last_mixture_weights = None

        nBatch = x.size(0)

        # Keep input dtype/device consistent with the stored normalization data.
        x = x.view(nBatch, -1).to(dtype=self.mean.dtype, device=self.mean.device)
        x0 = x * self.std + self.mean

        # Equivalent to exp(w_i) / sum_j exp(w_j), but numerically stable.
        weights = torch.softmax(self.wt, dim=0)
        u_sum = None
        for i in range(self.heads):
            u_i = self.sub_forward(x, x0, nBatch, sgn, i)
            u_sum = weights[i] * u_i if u_sum is None else u_sum + weights[i] * u_i

        if self.audit_enabled:
            self.last_mixture_weights = weights.detach().cpu().clone()

            # Margins are meaningful only for physical (unnormalized) controls.
            if (not self.training) and sgn == 0:
                theta1, w1 = x0[:, 0], x0[:, 1]
                theta2, w2 = x0[:, 2], x0[:, 3]
                phi_min = -np.pi * 2.2 / 3.0
                phi_max = np.pi * 2.2 / 3.0
                a_min, b_min = phi_min_ab(
                    theta1, w1, theta2, w2,
                    k1=2.0, k2=2.0, phi_min=phi_min,
                )
                a_max, b_max = phi_max_ab(
                    theta1, w1, theta2, w2,
                    k1=2.0, k2=2.0, phi_max=phi_max,
                )
                self.last_output_margins = {
                    "phi_min": hs_margin(u_sum, a_min, b_min).detach().cpu().clone(),
                    "phi_max": hs_margin(u_sum, a_max, b_max).detach().cpu().clone(),
                }

        return u_sum

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
    
    def _proj_tip_disk(self, u, theta1, w1, theta2, w2, k1_tip, k2_tip, return_ab: bool = False):
        """
        Tip disk CBF HOCBF (rel degree 2), same as你dCBF里tip那套
        Halfspace: a^T u >= b, where a = [Lg1, Lg2], b = -h
        """
        l1, l2 = 3.0, 3.0
        ox, oy, R = float(self.obs_x), float(self.obs_y), float(self.R)

        sint1 = torch.sin(theta1); cost1 = torch.cos(theta1)
        sint2 = torch.sin(theta2); cost2 = torch.cos(theta2)

        px = l1 * cost1 + l2 * cost2
        py = l1 * sint1 + l2 * sint2

        barrier = (px - ox) ** 2 + (py - oy) ** 2 - R ** 2
        barrier_dot = 2 * (px - ox) * (-l1 * sint1 * w1 - l2 * sint2 * w2) + \
                    2 * (py - oy) * ( l1 * cost1 * w1 + l2 * cost2 * w2)

        Lf2b = 2 * (-l1 * sint1 * w1 - l2 * sint2 * w2) ** 2 + \
            2 * ( l1 * cost1 * w1 + l2 * cost2 * w2) ** 2 + \
            2 * (px - ox) * (-l1 * cost1 * w1 ** 2 - l2 * cost2 * w2 ** 2) + \
            2 * (py - oy) * (-l1 * sint1 * w1 ** 2 - l2 * sint2 * w2 ** 2)

        # LgLf w.r.t [u1,u2] (== [theta1_ddot, theta2_ddot])
        Lg1 = 2 * (px - ox) * (-l1 * sint1) + 2 * (py - oy) * (l1 * cost1)
        Lg2 = 2 * (px - ox) * (-l2 * sint2) + 2 * (py - oy) * (l2 * cost2)

        h = Lf2b + (k1_tip + k2_tip) * barrier_dot + (k1_tip * k2_tip) * barrier

        a = torch.stack([Lg1, Lg2], dim=1)  # (B,2)
        b = (-h)                             # (B,)
        u2 = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u2, a, b, h
        return u2
    
    def _wrap_pi_torch(self, a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2.0 * torch.pi) - torch.pi

    def _proj_phi_min(self, u, theta1, w1, theta2, w2, k1=2.0, k2=2.0, phi_min=-2.0, return_ab: bool=False):
        phi = self._wrap_pi_torch(theta2 - theta1)
        phi_dot = w2 - w1

        barrier = phi - phi_min
        barrier_dot = phi_dot

        # Lf2b = 0 for double integrator in angles (no drift in accel)
        Lf2b = torch.zeros_like(phi)

        # LgLf for barrier: phi_ddot = u2 - u1
        Lg1 = -torch.ones_like(phi)  # coeff on u1
        Lg2 = +torch.ones_like(phi)  # coeff on u2

        h = Lf2b + (k1 + k2) * barrier_dot + (k1 * k2) * barrier

        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        u2 = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u2, a, b, h
        return u2


    def _proj_phi_max(self, u, theta1, w1, theta2, w2, k1=2.0, k2=2.0, phi_max=2.0, return_ab: bool=False):
        phi = self._wrap_pi_torch(theta2 - theta1)
        phi_dot = w2 - w1

        barrier = phi_max - phi
        barrier_dot = -phi_dot

        Lf2b = torch.zeros_like(phi)

        # now phi_ddot appears with opposite sign in barrier_ddot:
        # barrier_ddot = -(u2 - u1) = u1 - u2
        Lg1 = +torch.ones_like(phi)
        Lg2 = -torch.ones_like(phi)

        h = Lf2b + (k1 + k2) * barrier_dot + (k1 * k2) * barrier

        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        u2 = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u2, a, b, h
        return u2

    def dCBF_totalorder_proj(self, x0, x51, x52, sgn, nBatch, i):
        """
        Head i:
        x51 -> u_nom (physical)
        x52 -> kappa (tip+elbow)
        Then apply sequential projections following a fixed total order per-head.
        The current heads project tip first, then enumerate both orders of
        the incomparable co-maximal phi_min/phi_max pair.
        """

        # ---- states ----
        theta1 = x0[:, 0]; w1 = x0[:, 1]
        theta2 = x0[:, 2]; w2 = x0[:, 3]

        # ---- nominal u (physical) ----
        u = x51  # (B,2) physical

        # ---- per-head order ----
        order = self.orders[i % len(self.orders)]

        # ---- kappa from net (tip/elbow only) ----
        k1_tip = x52[:, 0]
        k2_tip = x52[:, 1]
        # k1_el  = x52[:, 2]
        # k2_el  = x52[:, 3]

        # ---- fixed phi gains (keep checkpoint unchanged) ----
        k1_phi, k2_phi = 2.0, 2.0
        phi_min, phi_max = -np.pi*2.2/3, np.pi*2.2/3

        # ---- sequential projection ----
        # Constraints already projected in this head
        seen_constraints = {}

        for name in order:
            u_before = u.clone()

            if name == "phi_min":
                u, a, b, _ = self._proj_phi_min(
                    u,
                    theta1,
                    w1,
                    theta2,
                    w2,
                    k1=k1_phi,
                    k2=k2_phi,
                    phi_min=phi_min,
                    return_ab=True,
                )

            elif name == "phi_max":
                u, a, b, _ = self._proj_phi_max(
                    u,
                    theta1,
                    w1,
                    theta2,
                    w2,
                    k1=k1_phi,
                    k2=k2_phi,
                    phi_max=phi_max,
                    return_ab=True,
                )

            elif name == "tip":
                u, a, b, _ = self._proj_tip_disk(
                    u,
                    theta1,
                    w1,
                    theta2,
                    w2,
                    k1_tip,
                    k2_tip,
                    return_ab=True,
                )

            else:
                raise ValueError(f"Unknown constraint name: {name}")

            u_after = u.clone()

            # Only phi_min and phi_max are incomparable and co-maximal.
            if self.audit_enabled:
                for previous_name, previous_data in seen_constraints.items():
                    is_comaximal_pair = {
                        previous_name,
                        name,
                    } == {
                        "phi_min",
                        "phi_max",
                    }

                    if not is_comaximal_pair:
                        continue

                    a_previous = previous_data["a"]
                    b_previous = previous_data["b"]

                    preserved_before = hs_margin(
                        u_before,
                        a_previous,
                        b_previous,
                    )

                    preserved_after = hs_margin(
                        u_after,
                        a_previous,
                        b_previous,
                    )

                    projected_before = hs_margin(
                        u_before,
                        a,
                        b,
                    )

                    projected_after = hs_margin(
                        u_after,
                        a,
                        b,
                    )

                    correction_norm = torch.linalg.vector_norm(
                        u_after - u_before,
                        dim=1,
                    )

                    cosine, intersection_slack = (
                        antiparallel_intersection_stats(
                            a_previous,
                            b_previous,
                            a,
                            b,
                        )
                    )

                    for batch_idx in range(nBatch):
                        self.last_audit.append({
                            "head": int(i),
                            "sample_index": int(batch_idx),
                            "order": tuple(order),
                            "preserved": previous_name,
                            "projected": name,
                            "preserved_margin_before": float(
                                preserved_before[batch_idx].detach().cpu()
                            ),
                            "preserved_margin_after": float(
                                preserved_after[batch_idx].detach().cpu()
                            ),
                            "projected_margin_before": float(
                                projected_before[batch_idx].detach().cpu()
                            ),
                            "projected_margin_after": float(
                                projected_after[batch_idx].detach().cpu()
                            ),
                            "correction_norm": float(
                                correction_norm[batch_idx].detach().cpu()
                            ),
                            "preserved_normal_norm": float(
                                torch.linalg.vector_norm(a_previous[batch_idx]).detach().cpu()
                            ),
                            "projected_normal_norm": float(
                                torch.linalg.vector_norm(a[batch_idx]).detach().cpu()
                            ),
                            "normal_cosine": float(
                                cosine[batch_idx].detach().cpu()
                            ),
                            "intersection_slack": float(
                                intersection_slack[batch_idx].detach().cpu()
                            ),
                        })

            seen_constraints[name] = {
                "a": a.detach().clone(),
                "b": b.detach().clone(),
            }

        if self.audit_enabled:
            self.last_head_outputs[int(i)] = u.detach().cpu().clone()

        # ---- output normalization behavior (match your original) ----
        if self.training or sgn == 1:
            u = (u - self.mean_label) / self.std_label

        return u


class PoSafeNet_Hard(nn.Module):
    def __init__(self, model_param, mean, std, mean_label, std_label, device, bn, heads = 2):
        super().__init__()
        self.bn = bn
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.mean_label = torch.from_numpy(mean_label).to(device)
        self.std_label = torch.from_numpy(std_label).to(device)
        self.device = device
        self.obs_x = 0  #obstacle location
        self.obs_y = 7
        self.R = 4   #obstacle size
        self.p1 = 0
        self.p2 = 0
        self.nCls = model_param[7]

        self.x52 = 0
        self.heads = heads

        self.add_noise = False

        # Audit-only state. These are plain Python attributes, so adding them
        # does not change checkpoint compatibility or trained parameters.
        self.audit_enabled = False
        self.last_audit = []
        self.last_head_outputs = {}
        self.last_output_margins = {}
        self.last_selected_head = None

        # self.orders = [
        #     ("phi_min", "phi_max", "tip",   "elbow"),
        #     ("phi_min", "phi_max", "elbow", "tip"),
        #     ("phi_max", "phi_min", "tip",   "elbow"),
        #     ("phi_max", "phi_min", "elbow", "tip"),
        # ]

        self.orders = [
            # ("phi_min", "phi_max", "elbow", "tip"),
            # ("phi_min", "phi_max", "elbow", "tip"),
            ("tip","phi_min", "phi_max"),
            # ("phi_max", "phi_min", "elbow", "tip"),
            # ("phi_max", "phi_min", "elbow", "tip"),
            ("tip","phi_max", "phi_min"),
        ]    
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(model_param[1])
            self.bn2 = nn.BatchNorm1d(model_param[2])
            self.bn31 = nn.BatchNorm1d(model_param[3])

        self.fc1, self.fc2, self.fc31, self.fc32, self.fc41, self.fc42, self.fc51, self.fc52 = nn.ModuleList(), nn.ModuleList(), \
              nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(self.heads):
            self.fc1.append(nn.Linear(model_param[0], model_param[1]).double().to(device))
            self.fc2.append(nn.Linear(model_param[1], model_param[2]).double().to(device))
            self.fc31.append(nn.Linear(model_param[2], model_param[3]).double().to(device))
            self.fc32.append(nn.Linear(model_param[2], model_param[4]).double().to(device))
            self.fc41.append(nn.Linear(model_param[3], model_param[5]).double().to(device))
            self.fc42.append(nn.Linear(model_param[4], model_param[6]).double().to(device))
            self.fc51.append(nn.Linear(model_param[5], model_param[7]).double().to(device))
            # self.fc52.append(nn.Linear(model_param[6], model_param[7]).double().to(device))
            self.fc52.append(nn.Linear(model_param[6], 2).double().to(device))

        self.activation = nn.ReLU()
        # self.activation = nn.Tanh()
        # QP params.
        # from previous layers
        self.wt = Parameter(torch.ones(self.heads))
    
    def sub_forward(self, x, x0, nBatch, sgn, i): 
        if self.add_noise:    
            x1 = self.activation(self.fc1[i](x + 0.1*torch.randn_like(x)))  #add noise to each model observation
        else:
            x1 = self.activation(self.fc1[i](x))
        if self.bn:
            x1 = self.bn1(x1)
        
        x2 = self.activation(self.fc2[i](x1))
        if self.bn:
            x2 = self.bn2(x2)

        x31 = self.activation(self.fc31[i](x2))
        x32 = self.activation(self.fc32[i](x2))

        x41 = self.activation(self.fc41[i](x31))
        x42 = self.activation(self.fc42[i](x32))

        x51 = self.fc51[i](x41)
        x52 = 4*nn.Sigmoid()(self.fc52[i](x42))  # ensure CBF parameters are positive

        if i == 0:
            self.x52 = x52
        
        # BarrierNet/ABNet
        u = self.dCBF_totalorder_proj(x0, x51, x52, sgn, nBatch, i)
        return u
    
    def forward(self, x, sgn, itr=0, tau=1.0, hard=True):
        if self.audit_enabled:
            self.last_audit = []
            self.last_head_outputs = {}
            self.last_output_margins = {}
            self.last_selected_head = None

        nBatch = x.size(0)

        # Keep dtype consistent with mean/std.
        x = x.view(nBatch, -1).to(dtype=self.mean.dtype, device=self.mean.device)
        x0 = x * self.std + self.mean

        # ----- inference: deterministic global head selection -----
        if not self.training:
            idx = int(torch.argmax(self.wt).item())
            u_out = self.sub_forward(x, x0, nBatch, sgn, idx)

            if self.audit_enabled:
                self.last_selected_head = idx

                # Margins are meaningful only for physical (unnormalized) controls.
                if sgn == 0:
                    theta1, w1 = x0[:, 0], x0[:, 1]
                    theta2, w2 = x0[:, 2], x0[:, 3]
                    phi_min = -np.pi * 2.2 / 3.0
                    phi_max = np.pi * 2.2 / 3.0
                    a_min, b_min = phi_min_ab(
                        theta1, w1, theta2, w2,
                        k1=2.0, k2=2.0, phi_min=phi_min,
                    )
                    a_max, b_max = phi_max_ab(
                        theta1, w1, theta2, w2,
                        k1=2.0, k2=2.0, phi_max=phi_max,
                    )
                    self.last_output_margins = {
                        "phi_min": hs_margin(u_out, a_min, b_min).detach().cpu().clone(),
                        "phi_max": hs_margin(u_out, a_max, b_max).detach().cpu().clone(),
                    }

            return u_out

        # ----- training: Gumbel-Softmax (Straight-Through if hard=True) -----
        y = F.gumbel_softmax(self.wt, tau=tau, hard=hard, dim=0)
        u_sel = None
        for i in range(self.heads):
            u_i = self.sub_forward(x, x0, nBatch, sgn, i)
            u_sel = y[i] * u_i if u_sel is None else u_sel + y[i] * u_i

        return u_sel

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
    
    def _proj_tip_disk(self, u, theta1, w1, theta2, w2, k1_tip, k2_tip, return_ab: bool = False):
        """
        Tip disk CBF HOCBF (rel degree 2), same as你dCBF里tip那套
        Halfspace: a^T u >= b, where a = [Lg1, Lg2], b = -h
        """
        l1, l2 = 3.0, 3.0
        ox, oy, R = float(self.obs_x), float(self.obs_y), float(self.R)

        sint1 = torch.sin(theta1); cost1 = torch.cos(theta1)
        sint2 = torch.sin(theta2); cost2 = torch.cos(theta2)

        px = l1 * cost1 + l2 * cost2
        py = l1 * sint1 + l2 * sint2

        barrier = (px - ox) ** 2 + (py - oy) ** 2 - R ** 2
        barrier_dot = 2 * (px - ox) * (-l1 * sint1 * w1 - l2 * sint2 * w2) + \
                    2 * (py - oy) * ( l1 * cost1 * w1 + l2 * cost2 * w2)

        Lf2b = 2 * (-l1 * sint1 * w1 - l2 * sint2 * w2) ** 2 + \
            2 * ( l1 * cost1 * w1 + l2 * cost2 * w2) ** 2 + \
            2 * (px - ox) * (-l1 * cost1 * w1 ** 2 - l2 * cost2 * w2 ** 2) + \
            2 * (py - oy) * (-l1 * sint1 * w1 ** 2 - l2 * sint2 * w2 ** 2)

        # LgLf w.r.t [u1,u2] (== [theta1_ddot, theta2_ddot])
        Lg1 = 2 * (px - ox) * (-l1 * sint1) + 2 * (py - oy) * (l1 * cost1)
        Lg2 = 2 * (px - ox) * (-l2 * sint2) + 2 * (py - oy) * (l2 * cost2)

        h = Lf2b + (k1_tip + k2_tip) * barrier_dot + (k1_tip * k2_tip) * barrier

        a = torch.stack([Lg1, Lg2], dim=1)  # (B,2)
        b = (-h)                             # (B,)
        u2 = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u2, a, b, h
        return u2
    
    def _wrap_pi_torch(self, a: torch.Tensor) -> torch.Tensor:
        return (a + torch.pi) % (2.0 * torch.pi) - torch.pi

    def _proj_phi_min(self, u, theta1, w1, theta2, w2, k1=2.0, k2=2.0, phi_min=-2.0, return_ab: bool=False):
        phi = self._wrap_pi_torch(theta2 - theta1)
        phi_dot = w2 - w1

        barrier = phi - phi_min
        barrier_dot = phi_dot

        # Lf2b = 0 for double integrator in angles (no drift in accel)
        Lf2b = torch.zeros_like(phi)

        # LgLf for barrier: phi_ddot = u2 - u1
        Lg1 = -torch.ones_like(phi)  # coeff on u1
        Lg2 = +torch.ones_like(phi)  # coeff on u2

        h = Lf2b + (k1 + k2) * barrier_dot + (k1 * k2) * barrier

        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        u2 = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u2, a, b, h
        return u2


    def _proj_phi_max(self, u, theta1, w1, theta2, w2, k1=2.0, k2=2.0, phi_max=2.0, return_ab: bool=False):
        phi = self._wrap_pi_torch(theta2 - theta1)
        phi_dot = w2 - w1

        barrier = phi_max - phi
        barrier_dot = -phi_dot

        Lf2b = torch.zeros_like(phi)

        # now phi_ddot appears with opposite sign in barrier_ddot:
        # barrier_ddot = -(u2 - u1) = u1 - u2
        Lg1 = +torch.ones_like(phi)
        Lg2 = -torch.ones_like(phi)

        h = Lf2b + (k1 + k2) * barrier_dot + (k1 * k2) * barrier

        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        u2 = self.proj_halfspace_batch(u, a, b)

        if return_ab:
            return u2, a, b, h
        return u2

    def dCBF_totalorder_proj(self, x0, x51, x52, sgn, nBatch, i):
        """
        Head i:
        x51 -> u_nom (physical)
        x52 -> kappa (tip+elbow)
        Then apply sequential projections following a fixed total order per-head.
        The current heads project tip first, then enumerate both orders of
        the incomparable co-maximal phi_min/phi_max pair.
        """

        # ---- states ----
        theta1 = x0[:, 0]; w1 = x0[:, 1]
        theta2 = x0[:, 2]; w2 = x0[:, 3]

        # ---- nominal u (physical) ----
        u = x51  # (B,2) physical

        # ---- per-head order ----
        order = self.orders[i % len(self.orders)]

        # ---- kappa from net (tip/elbow only) ----
        k1_tip = x52[:, 0]
        k2_tip = x52[:, 1]
        # k1_el  = x52[:, 2]
        # k2_el  = x52[:, 3]

        # ---- fixed phi gains (keep checkpoint unchanged) ----
        k1_phi, k2_phi = 2.0, 2.0
        phi_min, phi_max = -np.pi*2.2/3, np.pi*2.2/3

        # ---- sequential projection ----
        # Constraints already projected in this head
        seen_constraints = {}

        for name in order:
            u_before = u.clone()

            if name == "phi_min":
                u, a, b, _ = self._proj_phi_min(
                    u,
                    theta1,
                    w1,
                    theta2,
                    w2,
                    k1=k1_phi,
                    k2=k2_phi,
                    phi_min=phi_min,
                    return_ab=True,
                )

            elif name == "phi_max":
                u, a, b, _ = self._proj_phi_max(
                    u,
                    theta1,
                    w1,
                    theta2,
                    w2,
                    k1=k1_phi,
                    k2=k2_phi,
                    phi_max=phi_max,
                    return_ab=True,
                )

            elif name == "tip":
                u, a, b, _ = self._proj_tip_disk(
                    u,
                    theta1,
                    w1,
                    theta2,
                    w2,
                    k1_tip,
                    k2_tip,
                    return_ab=True,
                )

            else:
                raise ValueError(f"Unknown constraint name: {name}")

            u_after = u.clone()

            # Only phi_min and phi_max are incomparable and co-maximal.
            if self.audit_enabled:
                for previous_name, previous_data in seen_constraints.items():
                    is_comaximal_pair = {
                        previous_name,
                        name,
                    } == {
                        "phi_min",
                        "phi_max",
                    }

                    if not is_comaximal_pair:
                        continue

                    a_previous = previous_data["a"]
                    b_previous = previous_data["b"]

                    preserved_before = hs_margin(
                        u_before,
                        a_previous,
                        b_previous,
                    )

                    preserved_after = hs_margin(
                        u_after,
                        a_previous,
                        b_previous,
                    )

                    projected_before = hs_margin(
                        u_before,
                        a,
                        b,
                    )

                    projected_after = hs_margin(
                        u_after,
                        a,
                        b,
                    )

                    correction_norm = torch.linalg.vector_norm(
                        u_after - u_before,
                        dim=1,
                    )

                    cosine, intersection_slack = (
                        antiparallel_intersection_stats(
                            a_previous,
                            b_previous,
                            a,
                            b,
                        )
                    )

                    for batch_idx in range(nBatch):
                        self.last_audit.append({
                            "head": int(i),
                            "sample_index": int(batch_idx),
                            "order": tuple(order),
                            "preserved": previous_name,
                            "projected": name,
                            "preserved_margin_before": float(
                                preserved_before[batch_idx].detach().cpu()
                            ),
                            "preserved_margin_after": float(
                                preserved_after[batch_idx].detach().cpu()
                            ),
                            "projected_margin_before": float(
                                projected_before[batch_idx].detach().cpu()
                            ),
                            "projected_margin_after": float(
                                projected_after[batch_idx].detach().cpu()
                            ),
                            "correction_norm": float(
                                correction_norm[batch_idx].detach().cpu()
                            ),
                            "preserved_normal_norm": float(
                                torch.linalg.vector_norm(a_previous[batch_idx]).detach().cpu()
                            ),
                            "projected_normal_norm": float(
                                torch.linalg.vector_norm(a[batch_idx]).detach().cpu()
                            ),
                            "normal_cosine": float(
                                cosine[batch_idx].detach().cpu()
                            ),
                            "intersection_slack": float(
                                intersection_slack[batch_idx].detach().cpu()
                            ),
                        })

            seen_constraints[name] = {
                "a": a.detach().clone(),
                "b": b.detach().clone(),
            }

        if self.audit_enabled:
            self.last_head_outputs[int(i)] = u.detach().cpu().clone()

        # ---- output normalization behavior (match your original) ----
        if self.training or sgn == 1:
            u = (u - self.mean_label) / self.std_label

        return u

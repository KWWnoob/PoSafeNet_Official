import json
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch
from .base import LitModel as Base
from .utils import build_cnn, build_mlp
import numpy as np
import torch.nn.functional as F

class LitModel(Base):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = Base.add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group('models.barrier_net')
        parser.add_argument('--cnn-params', type=json.loads,
                            default=[[3, 24, 5, 2, 2],
                                     [24, 36, 5, 2, 2],
                                     [36, 48, 3, 2, 1],
                                     [48, 64, 3, 1, 1],
                                     [64, 64, 3, 1, 1]])
        parser.add_argument('--cnn-dropout', type=float, default=0.3)
        parser.add_argument('--cnn-with-gn', action='store_true', default=False)
        parser.add_argument('--cnn-activation', type=str, default='relu')
        parser.add_argument('--keep-w-feature', action='store_true', default=False)
        parser.add_argument('--lstm-size', type=int, default=64)
        parser.add_argument('--p-mlp-params', type=json.loads,
                            default=[32, 32, 2])
        parser.add_argument('--p-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--p-mlp-activation', type=str, default='relu')
        parser.add_argument('--p-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--p-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--p-mlp-no-output-norm', action='store_true', default=False)
        parser.add_argument('--p-lower-bound', type=float, default=None)
        parser.add_argument('--q-mlp-params', type=json.loads,
                            default=[32, 32, 2])
        parser.add_argument('--q-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--q-mlp-activation', type=str, default='relu')
        parser.add_argument('--q-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--q-mlp-with-gn', action='store_true', default=False)
        parser.add_argument('--model-type', type=str, default='deri',
                            choices=['deri', 'inte', 'direct'])
        parser.add_argument('--not-use-gt', action='store_true', default=False)
        parser.add_argument('--randomize-q', action='store_true', default=False)
        parser.add_argument('--detach-q', action='store_true', default=False)
        parser.add_argument('--clip-q', action='store_true', default=False)

        parser.add_argument('--use-state-net', action='store_true', default=False)
        parser.add_argument('--use-indep-state-cnn', action='store_true', default=False)
        parser.add_argument('--state-tol', type=float, default=0.05)
        parser.add_argument('--use-lstm-in-state-net', action='store_true', default=False)
        parser.add_argument('--ds-mlp-params', type=json.loads, default=None)
        parser.add_argument('--ds-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--ds-mlp-activation', type=str, default='relu')
        parser.add_argument('--ds-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--dd-mlp-params', type=json.loads, default=None)
        parser.add_argument('--dd-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--dd-mlp-activation', type=str, default='relu')
        parser.add_argument('--dd-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--d-mlp-params', type=json.loads, default=None)
        parser.add_argument('--d-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--d-mlp-activation', type=str, default='relu')
        parser.add_argument('--d-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--mu-mlp-params', type=json.loads, default=None)
        parser.add_argument('--mu-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--mu-mlp-activation', type=str, default='relu')
        parser.add_argument('--mu-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--kappa-mlp-params', type=json.loads, default=None)
        parser.add_argument('--kappa-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--kappa-mlp-activation', type=str, default='relu')
        parser.add_argument('--kappa-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--obs-d-mlp-params', type=json.loads, default=None)
        parser.add_argument('--obs-d-mlp-with-bn', action='store_true', default=False)
        parser.add_argument('--obs-d-mlp-activation', type=str, default='relu')
        parser.add_argument('--obs-d-mlp-dropout', type=float, default=0.3)
        parser.add_argument('--use-lane-keeping-CBFs', action='store_true', default=False)
        parser.add_argument('--abnet-heads', type=int, default=10)
        parser.add_argument('--use-attention', action='store_true', default=False)
        parser.add_argument('--use-cf', action='store_true', default=False)

        parser.add_argument(
            '--learn-mixture-weight',
            action='store_true',
            default=False,
            help='Whether to learn state-dependent mixture weights over abnet heads'
        )
                
        return parent_parser

    def custom_setup(self):
        if self.hparams.model_type == 'deri':
            assert self.hparams.output_mode[:4] == ['v', 'delta', 'a', 'omega'], \
                'Type deri of barrier net uses (v, delta, a, omega) as output'
        elif self.hparams.model_type in ['inte', 'direct']:
            assert self.hparams.output_mode[:2] == ['a', 'omega'], \
                'Type inte/direct of barrier net uses (a, omega) as output'
        self._validify_loss_ceof() 

        if self.hparams.get('keep_w_feature', False):
            cnn_feat_size = self.hparams.cnn_params[-1][0] * 20 # NOTE: hacky way to get feature w
        else:
            cnn_feat_size = self.hparams.cnn_params[-1][0]
        
        self._cnn, self._lstm, self._p_mlp, self._q_mlp = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for k in range(self.hparams.abnet_heads):
            self._cnn.append(build_cnn(filters=self.hparams.cnn_params,
                                dropout=self.hparams.cnn_dropout,
                                activation=self.hparams.get('cnn_activation', 'relu'),
                                with_gn=self.hparams.get('cnn_with_gn', False)))

            
            self._lstm.append(nn.LSTM(cnn_feat_size, self.hparams.lstm_size, batch_first=True))

            p_mlp_params = [self.hparams.lstm_size] + self.hparams.p_mlp_params
            self._p_mlp.append(build_mlp(filters=p_mlp_params,
                                    dropout=self.hparams.p_mlp_dropout,
                                    with_bn=self.hparams.p_mlp_with_bn,
                                    with_gn=self.hparams.get('p_mlp_with_gn', False),
                                    with_output_norm=not self.hparams.get('p_mlp_no_output_norm', False),
                                    no_act_last_layer=True,
                                    activation=self.hparams.p_mlp_activation))

            q_mlp_params = [self.hparams.lstm_size] + self.hparams.q_mlp_params
            self._q_mlp.append(build_mlp(filters=q_mlp_params,
                                    dropout=self.hparams.q_mlp_dropout,
                                    with_bn=self.hparams.q_mlp_with_bn,
                                    with_gn=self.hparams.get('p_mlp_with_gn', False),
                                    no_act_last_layer=True,
                                    activation=self.hparams.q_mlp_activation))

        

        if self.hparams.get('use_state_net', False):
            assert self.hparams.output_mode[-3:] == ['ds', 'dd', 'mu'] or \
                   self.hparams.output_mode[-4:] == ['ds', 'obs_d', 'd', 'mu'] or \
                   self.hparams.output_mode[-5:] == ['ds', 'obs_d', 'd', 'mu', 'kappa']

            self.hparams.ds_bound = [float(v) for v in self.hparams.ds_bound]
            if self.hparams.ds_bound_for_obs_d:
                self.hparams.ds_bound_for_obs_d = [float(v) for v in self.hparams.ds_bound_for_obs_d]

            if self.hparams.use_indep_state_cnn:
                self._state_cnn = build_cnn(filters=self.hparams.cnn_params,
                                            dropout=self.hparams.cnn_dropout)

                if self.hparams.keep_w_feature:
                    cnn_feat_size = self.hparams.cnn_params[-1][0] * 20 # NOTE: hacky way to get feature w
                else:
                    cnn_feat_size = self.hparams.cnn_params[-1][0]
                self._state_lstm = nn.LSTM(cnn_feat_size,
                                        self.hparams.lstm_size,
                                        batch_first=True)

            feat_size = self.hparams.lstm_size if \
                self.hparams.use_lstm_in_state_net else cnn_feat_size

            if self.hparams.ds_mlp_params:
                self._ds_mlp = build_mlp(filters=[feat_size] + self.hparams.ds_mlp_params,
                                         dropout=self.hparams.ds_mlp_dropout,
                                         with_bn=self.hparams.ds_mlp_with_bn,
                                         no_act_last_layer=True,
                                         activation=self.hparams.ds_mlp_activation)

            if self.hparams.dd_mlp_params:
                self._dd_mlp = build_mlp(filters=[feat_size] + self.hparams.dd_mlp_params,
                                         dropout=self.hparams.dd_mlp_dropout,
                                         with_bn=self.hparams.dd_mlp_with_bn,
                                         no_act_last_layer=True,
                                         activation=self.hparams.dd_mlp_activation)

            if self.hparams.d_mlp_params:
                self._d_mlp = build_mlp(filters=[feat_size] + self.hparams.d_mlp_params,
                                        dropout=self.hparams.d_mlp_dropout,
                                        with_bn=self.hparams.d_mlp_with_bn,
                                        no_act_last_layer=True,
                                        activation=self.hparams.d_mlp_activation)

            if self.hparams.mu_mlp_params:
                self._mu_mlp = build_mlp(filters=[feat_size] + self.hparams.mu_mlp_params,
                                         dropout=self.hparams.mu_mlp_dropout,
                                         with_bn=self.hparams.mu_mlp_with_bn,
                                         no_act_last_layer=True,
                                         activation=self.hparams.mu_mlp_activation)

            if self.hparams.kappa_mlp_params:
                self._kappa_mlp = build_mlp(filters=[feat_size] + self.hparams.kappa_mlp_params,
                                            dropout=self.hparams.kappa_mlp_dropout,
                                            with_bn=self.hparams.kappa_mlp_with_bn,
                                            no_act_last_layer=True,
                                            activation=self.hparams.kappa_mlp_activation)

            if self.hparams.obs_d_mlp_params:
                self._obs_d_mlp = build_mlp(filters=[feat_size] + self.hparams.obs_d_mlp_params,
                                            dropout=self.hparams.obs_d_mlp_dropout,
                                            with_bn=self.hparams.obs_d_mlp_with_bn,
                                            no_act_last_layer=True,
                                            activation=self.hparams.obs_d_mlp_activation)
        self.wt = Parameter(torch.ones(self.hparams.abnet_heads))  #weights on bnet
        self.p = 0    #inter-connected penalty
        self.barrier = 0


        if self.hparams.get('use_attention', False):
            attention_point = [0, 0, 0, 0, 38, 38, 77, 115, 115, 154]
            factor = [1.0, 1.0, 1.0, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
            self.attention_map = []
            for k in range(self.hparams.abnet_heads):
                self.attention_map.append(self.generate_attention_matrix(45, 155, 22, attention_point[k], decay_factor = factor[k]).unsqueeze(0).expand(3, 45, 155))

        self.learn_mixture_weight = self.hparams.get('learn_mixture_weight', False)
        self.wt.requires_grad_(self.learn_mixture_weight)

            
    def generate_attention_matrix(self, h, w, fx, fy, decay_factor = 0.95):
        att_matrix = torch.zeros(h, w)
        for i in range(h):
            for j in range(w):
                # att_matrix[i,j] = decay_factor**torch.max(torch.tensor([np.abs(fx - i), np.abs(fy - j)]))
                att_matrix[i,j] = decay_factor**torch.max(torch.tensor(np.sqrt([(np.float(fx) - np.float(i))**2 + (np.float(fy) - np.float(j))**2])))
                # att_matrix[i,j] = torch.exp(-decay_factor**torch.max(torch.tensor(np.sqrt([(np.float(fx) - np.float(i))**2 + (np.float(fy) - np.float(j))**2]))))
        return att_matrix

    # @staticmethod
    def proj_halfspace_batch(self, u: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
        """ 
        u: (B,m)
        a: (B,m)
        b: (B,) or (B,1)
        projects each row u_i onto {u | a_i^T u >= b_i}
        """
        if b.ndim == 2 and b.shape[1] == 1:
            b = b[:, 0]
        # align dtype/device
        a = a.to(dtype=u.dtype, device=u.device)
        b = b.to(dtype=u.dtype, device=u.device)

        au = (a * u).sum(dim=1)                      # (B,)
        viol = b - au                                # (B,)
        denom = (a * a).sum(dim=1) + eps             # (B,)
        step = torch.relu(viol) / denom              # (B,)
        return u + step[:, None] * a                 # (B,m)

    # ----------------------------
    # 3 atomic projections
    # ----------------------------
    def _proj_lane_left(self, u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf):
        lf = self.hparams.get('lf_cbf_threshold', 2.0)

        barrier = lf - d
        barrier_dot = -v * sin_mu_beta
        Lf2b = -v * cos_mu_beta * mu_dot

        Lg1 = -sin_mu_beta
        Lg2 = -v * cos_mu_beta * lrf / torch.cos(delta)**2 / (1 + (lrf * torch.tan(delta))**2)

        # your original lane keeping HOCBF form:
        h = Lf2b + 2.0 * barrier_dot + 1.0 * barrier

        a = torch.stack([Lg1, Lg2], dim=1)   # (B,2)
        b = (-h)                              # (B,)
        return self.proj_halfspace_batch(u, a, b)

    def _proj_lane_right(self, u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf):
        lf = self.hparams.get('lf_cbf_threshold', 2.0)

        barrier = d + lf
        barrier_dot = v * sin_mu_beta
        Lf2b = v * cos_mu_beta * mu_dot

        Lg1 = sin_mu_beta
        Lg2 = v * cos_mu_beta * lrf / torch.cos(delta)**2 / (1 + (lrf * torch.tan(delta))**2)

        h = Lf2b + 2.0 * barrier_dot + 1.0 * barrier

        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        return self.proj_halfspace_batch(u, a, b)
    
    def _proj_center_left(self, u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf):
        tol = float(self.hparams.get('centerline_tol', 0.30))
        k1 = float(self.hparams.get('centerline_k1', 2.0))
        k0 = float(self.hparams.get('centerline_k0', 1.0))

        barrier = tol - d
        barrier_dot = -v * sin_mu_beta
        Lf2b = -v * cos_mu_beta * mu_dot
        Lg1 = -sin_mu_beta
        Lg2 = -v * cos_mu_beta * lrf / torch.cos(delta)**2 / (1 + (lrf * torch.tan(delta))**2)

        h = Lf2b + k1 * barrier_dot + k0 * barrier
        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        return self.proj_halfspace_batch(u, a, b)

    def _proj_center_right(self, u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf):
        tol = float(self.hparams.get('centerline_tol', 0.30))
        k1 = float(self.hparams.get('centerline_k1', 2.0))
        k0 = float(self.hparams.get('centerline_k0', 1.0))

        barrier = d + tol
        barrier_dot = v * sin_mu_beta
        Lf2b = v * cos_mu_beta * mu_dot
        Lg1 = sin_mu_beta
        Lg2 = v * cos_mu_beta * lrf / torch.cos(delta)**2 / (1 + (lrf * torch.tan(delta))**2)

        h = Lf2b + k1 * barrier_dot + k0 * barrier
        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        return self.proj_halfspace_batch(u, a, b)

    def _proj_obs_disk(self, u, ds, dd, d, v, delta, kappa, cos_mu_beta, sin_mu_beta, mu_dot, lrf, p, iter):
        # obstacle disk barrier (same as your dCBF)
        barrier = ds**2 + dd**2 - 7.9**2
        barrier_dot = 2*ds*v*cos_mu_beta/(1 - d*kappa) + 2*dd*v*sin_mu_beta

        Lf2b = 2*(v*cos_mu_beta/(1 - d*kappa))**2 + 2*(v*sin_mu_beta)**2 \
            - 2*ds*v*sin_mu_beta*mu_dot/(1 - d*kappa) \
            + 2*ds*kappa*v**2*sin_mu_beta*cos_mu_beta/(1 - d*kappa)**2 \
            + 2*dd*v*cos_mu_beta*mu_dot

        Lg1 = 2*ds*cos_mu_beta/(1 - d*kappa) + 2*dd*sin_mu_beta
        Lg2 = (-2*ds*v*sin_mu_beta/(1 - d*kappa) + 2*dd*v*cos_mu_beta) * lrf / torch.cos(delta)**2 / (1 + (lrf*torch.tan(delta))**2)

        # keep your per-head p-coupling logic EXACTLY
        if iter == 0:
            k0 = p[:, 0]
            k1 = p[:, 1]
        else:
            k0 = self.p[:, 0]
            k1 = p[:, 1]

        h = Lf2b + (k0 + k1) * barrier_dot + (k0 * k1) * barrier

        a = torch.stack([Lg1, Lg2], dim=1)
        b = (-h)
        return self.proj_halfspace_batch(u, a, b)

    def _proj_speed_max(self, u, v):
        # enforce v <= v_max assuming v_dot = a and u[:,1] = a
        v_max = float(self.hparams.get('v_max', 7.0))
        k = float(self.hparams.get('speed_k', 1.0))

        # a <= k (v_max - v)  <=>  -a >= -k (v_max - v)
        a_vec = torch.stack([torch.zeros_like(v), -torch.ones_like(v)], dim=1)  # (B,2)
        b = -k * (v_max - v)                                                   # (B,)
        return self.proj_halfspace_batch(u, a_vec, b)

    def _proj_speed_min(self, u, v):
        # enforce v >= v_min assuming v_dot = a and u[:,1] = a
        v_min = float(self.hparams.get('v_min', 0.0))
        k = float(self.hparams.get('speed_k', 1.0))

        # a >= -k (v - v_min)
        a_vec = torch.stack([torch.zeros_like(v), torch.ones_like(v)], dim=1)   # (B,2)
        b = -k * (v - v_min)                                                   # (B,)
        return self.proj_halfspace_batch(u, a_vec, b)
    
    # ----------------------------
    # sequential version of dCBF (ONLY: obs_disk, lane_left, lane_right)
    # with head-specific lane swap:
    #   head even:  L -> R -> obs
    #   head odd:   R -> L -> obs
    # ----------------------------
    def dCBF_seq(self, nBatch, batch, state_seq, obs_seq, img_seq_fl, b, t, z, p, q, solver, iter: int):
        """
        Sequential projection controller:
          nominal u = -q
          lane constraints first (low priority) with head-dependent swap
          obstacle disk last (high priority)
        """

        # ===========================
        # COPY of your big "state_net / gt" parsing logic
        # (verbatim from dCBF, but no QP construction)
        # ===========================

        gt_s, gt_d, gt_mu, v, delta, gt_kappa = state_seq[:,0], state_seq[:,1], \
                state_seq[:,2], state_seq[:,3], state_seq[:,4], state_seq[:,5]
        gt_obs_s, gt_obs_d = obs_seq[:,0], obs_seq[:,1]
        gt_ds = gt_s - gt_obs_s
        gt_dd = gt_d - gt_obs_d

        if self.hparams.get('use_state_net', False):
            if self.hparams.use_indep_state_cnn:
                state_z = self._state_cnn(img_seq_fl)
                if self.hparams.keep_w_feature:
                    state_z = state_z.max(-2)[0]  # max-pooling over h dim only
                    state_z = state_z.flatten(-2, -1)  # flatten c and h
                else:
                    state_z = state_z.max(-1)[0].max(-1)[0]  # max-pooling over h and w dims
                cnn_feat = state_z
                state_z = state_z.view(b, t, -1)
                state_z, rnn_state = self._lstm(state_z, rnn_state)
            else:
                state_z = z

            state_net_in = state_z if self.hparams.use_lstm_in_state_net else cnn_feat
            d = self._d_mlp(state_net_in).squeeze(1) if self.hparams.d_mlp_params else gt_d
            mu = self._mu_mlp(state_net_in).squeeze(1) if self.hparams.mu_mlp_params else gt_mu
            kappa = self._kappa_mlp(state_net_in).squeeze(1) if self.hparams.kappa_mlp_params else gt_kappa

            obs_d = self._obs_d_mlp(state_net_in).squeeze(1) if self.hparams.d_mlp_params else gt_obs_d
            if self.hparams.drop_obs_d_offset:
                obs_d = obs_d + torch.sign(obs_d) * 5
            ds = self._ds_mlp(state_net_in).squeeze(1) if self.hparams.ds_mlp_params else gt_s - gt_obs_s
            dd = self._dd_mlp(state_net_in).squeeze(1) if self.hparams.dd_mlp_params else d - obs_d
            if self.hparams.drop_obs_d_offset:
                dd = dd + torch.sign(dd) * 5

            # keep non_detach vars if you later need them
            d_non_detach = d
            mu_non_detach = mu
            kappa_non_detach = kappa
            obs_d_non_detach = obs_d
            ds_non_detach = ds
            dd_non_detach = dd

            tol = self.hparams.state_tol
            d = torch.clamp(d.detach(), gt_d * (1 - tol), gt_d * (1 + tol))
            mu = torch.clamp(mu.detach(), gt_mu * (1 - tol), gt_mu * (1 + tol))
            kappa = torch.clamp(kappa.detach(), gt_kappa * (1 - tol), gt_kappa * (1 + tol))
            obs_d = torch.clamp(obs_d.detach(), gt_obs_d * (1 - tol), gt_obs_d * (1 + tol))
            ds = torch.clamp(ds.detach(), gt_ds * (1 - tol), gt_ds * (1 + tol))
            dd = torch.clamp(dd.detach(), gt_dd * (1 - tol), gt_dd * (1 + tol))

        elif self.hparams.get('use_indep_state_net', False):
            indep_state_pred = batch[4].reshape(b * t, -1)
            if self.hparams.indep_state_net_output == ['ds', 'd', 'mu', 'kappa', 'obs_d']:
                ds = indep_state_pred[:,0]
                d = indep_state_pred[:,1]
                mu = indep_state_pred[:,2]
                kappa = indep_state_pred[:,3]
                obs_d = indep_state_pred[:,4]
                dd = d - obs_d
            elif self.hparams.indep_state_net_output == ['ds', 'obs_d', 'd', 'mu']:
                ds = indep_state_pred[:,0]
                obs_d = indep_state_pred[:,1]
                d = indep_state_pred[:,2]
                mu = indep_state_pred[:,3]
                kappa = torch.zeros_like(gt_kappa)
                dd = d - obs_d
            elif self.hparams.indep_state_net_output == ['ds', 'dd', 'mu']:
                ds = indep_state_pred[:,0]
                dd = indep_state_pred[:,1]
                mu = indep_state_pred[:,2]
                kappa = torch.zeros_like(gt_kappa)
                d = gt_d
            else:
                raise NotImplementedError(f'Unrecognized state net output {self.hparams.indep_state_net_output}')

        else:
            # pure GT
            d, mu, kappa = gt_d, gt_mu, gt_kappa
            ds, dd = gt_ds, gt_dd

        self.barrier = ds**2 + dd**2 - 7.9**2 #barrire
        # ===========================
        # dynamics terms (same as dCBF)
        # ===========================
        lrf, lr = 0.5, 2.78
        beta = torch.atan(lrf * torch.tan(delta))
        cos_mu_beta = torch.cos(mu + beta)
        sin_mu_beta = torch.sin(mu + beta)
        mu_dot = v/lr * torch.sin(beta) - kappa*v*cos_mu_beta/(1 - d*kappa)

        # ===========================
        # sequential projection
        # ===========================
        u = (-q).clone()   # nominal (same as QP unconstrained optimum)

        # lane swap across heads:
        # even head: L then R; odd head: R then L
        head_id = int(iter) % 4
        lane_L_then_R = (head_id in [0, 2])   # 0/2: L->R, 1/3: R->L
        speed_max_first = (head_id in [0, 1]) # 0/1: max->min, 2/3: min->max

        if lane_L_then_R:
            u = self._proj_center_left(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)
            u = self._proj_center_right(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)
        else:
            u = self._proj_center_right(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)
            u = self._proj_center_left(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)

        if speed_max_first:
            u = self._proj_speed_max(u, v)
            u = self._proj_speed_min(u, v)
        else:
            u = self._proj_speed_min(u, v)
            u = self._proj_speed_max(u, v)

        if lane_L_then_R:
            u = self._proj_lane_left(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)
            u = self._proj_lane_right(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)
        else:
            u = self._proj_lane_right(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)
            u = self._proj_lane_left(u, d, v, delta, cos_mu_beta, sin_mu_beta, mu_dot, lrf)

        # obstacle ALWAYS last (highest priority)
        u = self._proj_obs_disk(u, ds, dd, d, v, delta, kappa, cos_mu_beta, sin_mu_beta, mu_dot, lrf, p, iter)

        return u


    def sub_forward(self, batch, rnn_state, iter, solver):

        img_seq = batch[0]
        state_seq = batch[1]
        obs_seq = batch[2]
        ctrl_seq = batch[3]

        b, t, c, h, w = img_seq.shape 
        img_seq_fl = img_seq.view(b * t, c, h, w)  # 3 x 45 x 155

        # import matplotlib.image as pim  # NOTE: test attention observation
        # mx = self.generate_attention_matrix(45, 155, 22, 0, decay_factor = 0.95)
        # img = np.moveaxis((img_seq_fl[0]*mx.unsqueeze(0).expand(c, h, w).to(img_seq.device)).cpu().numpy(), 0, -1)
        # pim.imsave('test1.png', img)

        if self.hparams.get('use_attention', False):
            img_seq_fl = img_seq_fl*self.attention_map[iter].unsqueeze(0).expand(b * t, c, h, w).to(img_seq.device)

        z = self._cnn[iter](img_seq_fl)  # NOTE: add noise    + 0.5*torch.randn_like(img_seq_fl)
        if self.hparams.get('keep_w_feature', False):
            z = z.max(-2)[0] # max-pooling over h dim only
            z = z.flatten(-2, -1) # flatten c and h
        else:
            z = z.max(-1)[0].max(-1)[0] # max-pooling over h and w dims
        cnn_feat = z
        z = z.view(b, t, -1)

        z, rnn_state = self._lstm[iter](z, rnn_state)

        z = z.reshape(b * t, -1)
        q = self._q_mlp[iter](z)
        p = self._p_mlp[iter](z)*3

        if iter == 0:
            self.p = p

        if self.hparams.p_lower_bound is not None:
            p = torch.clamp(p, min=self.hparams.p_lower_bound)

        if solver == 'cvxpy':
            q = q.view(b, t, -1)
            q = q[:,-1]
            p = p.view(b, t, -1)
            p = p[:,-1]
            state_seq = state_seq[:,-1]
            obs_seq = obs_seq[:,-1]
            ctrl_seq = ctrl_seq[:,-1]
            nBatch = b
        else:
            nBatch = b*t
            state_seq = state_seq.view(nBatch, -1)
            obs_seq = obs_seq.view(nBatch, -1)
            ctrl_seq = ctrl_seq.view(nBatch, -1)

        #softened HOCBFs for safety
        if self.hparams.model_type in ['deri', 'deri_ref']:
            v_delta = q
            if solver == 'cvxpy' or self.hparams.not_use_gt:
                q = -((q - state_seq[:,3:5])/0.1)

                if self.hparams.detach_q:
                    q = q.detach()

                if self.hparams.clip_q or solver == 'cvxpy': # NOTE: bound q during eval
                    q[:, 0] = torch.clamp(q[:, 0], -1.0, 1.0) # -omega
                    q[:, 1] = torch.clamp(q[:, 1], -7.0, 7.0) # -a
            elif self.hparams.randomize_q and self.training:
                q = torch.stack([(torch.rand(q[:, 0].shape) - 0.5) * 1,
                                 (torch.rand(q[:, 1].shape) - 0.5) * 7], dim=1).to(q)
            else:
                q = -ctrl_seq[:,2:4]
        elif self.hparams.model_type in ['inte', 'direct']:
            pass # reference control q is a-omega
        
        if self.hparams.get('use_cf', False):
            # PoSafeNet: sequential closed-form half-space projection
            x = self.dCBF_seq(nBatch, batch, state_seq, obs_seq, img_seq_fl, b, t, z, p, q, solver, iter)
        else:
            raise ValueError('PoSafeNet requires --use-cf (sequential closed-form projection).')
  
        
        return x, rnn_state, v_delta
    
    def forward(self, batch, rnn_state, solver='QP', store_intermediate_data=False, 
            log_head_disagreement=False, **kwargs):
        b, t, c, h, w = batch[0].shape

        rnn_state_rt = []
        v_delta = []
        x = 0

        K = self.hparams.abnet_heads
        learn_w = self.learn_mixture_weight

        if learn_w:
            alpha = torch.softmax(self.wt, dim=0)
        else:
            alpha = torch.full((K,), 1.0 / K, device=self.wt.device, dtype=self.wt.dtype)

        # ========== 新增：收集每个 head 的 output ==========
        head_outputs = []
        # ==================================================

        for k in range(K):
            x_k, rnn_state_k, v_delta_k = self.sub_forward(batch, rnn_state[k], k, solver)
            x = x + x_k * alpha[k]
            rnn_state_rt.append(rnn_state_k)
            v_delta.append(v_delta_k.unsqueeze(0))

            # ========== 新增：存每个 head 的 raw output ==========
            if log_head_disagreement:
                head_outputs.append(x_k.detach().clone())
            # ====================================================

        v_delta = torch.mean(torch.cat(v_delta, dim=0), dim=0)
        rnn_state = rnn_state_rt

        if self.hparams.model_type == 'deri':
            out = torch.cat([v_delta, x], dim=1)
        elif self.hparams.model_type == 'direct':
            out = x

        out = out.view(b, t, -1)

        # ========== 新增：计算并存储 head disagreement ==========
        if log_head_disagreement:
            stacked = torch.stack(head_outputs, dim=0)  # (K, B*T, 2)
            # max pairwise L2 distance across heads, per sample
            max_disagree = torch.zeros(stacked.shape[1], device=stacked.device)
            for i in range(K):
                for j in range(i+1, K):
                    dist = torch.norm(stacked[i] - stacked[j], dim=-1)  # (B*T,)
                    max_disagree = torch.max(max_disagree, dist)
            
            self._head_log = {
                'head_actions': stacked.cpu().numpy(),       # (K, B*T, 2)
                'max_disagreement': max_disagree.cpu().numpy(),  # (B*T,)
                'alpha': alpha.detach().cpu().numpy(),        # (K,)
                'final_action': x.detach().cpu().numpy(),    # (B*T, 2)
                'barrier_value': self.barrier.detach().cpu().numpy() if hasattr(self, 'barrier') else None,
            }
        # ======================================================

        return out, rnn_state

    # def forward(self, batch, rnn_state, solver='QP', store_intermediate_data=False,
    #         tau: float = 1.0, hard: bool = True, **kwargs):
    #     """
    #     Global head gating with (optional) Gumbel-Softmax.

    #     - training:
    #         y = gumbel_softmax(wt)  # (K,)
    #         x = Σ_k y[k] * x_k      # differentiable; hard=True uses straight-through
    #     - eval:
    #         pick k* = argmax(wt) and only run that head
    #     """

    #     b, t, c, h, w = batch[0].shape
    #     K = self.hparams.abnet_heads
    #     learn_w = self.learn_mixture_weight

    #     # ---------- deterministic weights if not learning ----------
    #     # if learn_w==False: uniform mixing (your original)
    #     if not learn_w:
    #         alpha = torch.full((K,), 1.0 / K, device=self.wt.device, dtype=self.wt.dtype)

    #         rnn_state_rt = []
    #         v_delta_list = []
    #         x = None

    #         for k in range(K):
    #             x_k, rnn_state_k, v_delta_k = self.sub_forward(batch, rnn_state[k], k, solver)
    #             x = x_k * alpha[k] if x is None else x + x_k * alpha[k]
    #             rnn_state_rt.append(rnn_state_k)
    #             v_delta_list.append(v_delta_k.unsqueeze(0))

    #         v_delta = torch.mean(torch.cat(v_delta_list, dim=0), dim=0)
    #         rnn_state = rnn_state_rt

    #     # ---------- learnable gating ----------
    #     else:
    #         if not self.training:
    #             k_star = int(torch.argmax(self.wt).item())

    #             # warmup head0 to refresh self.p (batch-aligned)
    #             if k_star != 0:
    #                 _x0, rnn0, _vd0 = self.sub_forward(batch, rnn_state[0], 0, solver)
    #                 rnn_state = list(rnn_state)
    #                 rnn_state[0] = rnn0

    #             # run selected head
    #             x, rnnk, v_delta = self.sub_forward(batch, rnn_state[k_star], k_star, solver)
    #             rnn_state = list(rnn_state)
    #             rnn_state[k_star] = rnnk

    #         else:
    #             # training: gumbel selection (global, same for all samples)
    #             # self.wt is logits (K,)
    #             y = F.gumbel_softmax(self.wt, tau=tau, hard=hard, dim=0)  # (K,)

    #             rnn_state_rt = []
    #             v_delta_list = []
    #             x = None

    #             for k in range(K):
    #                 x_k, rnn_state_k, v_delta_k = self.sub_forward(batch, rnn_state[k], k, solver)
    #                 x = x_k * y[k] if x is None else x + x_k * y[k]
    #                 rnn_state_rt.append(rnn_state_k)
    #                 v_delta_list.append(v_delta_k.unsqueeze(0))

    #             # keep your original v_delta behavior: average across heads
    #             v_delta = torch.mean(torch.cat(v_delta_list, dim=0), dim=0)
    #             rnn_state = rnn_state_rt

    #     # ---------- build output (same as your current forward) ----------
    #     if self.hparams.model_type == 'deri':
    #         out = torch.cat([v_delta, x], dim=1)
    #     elif self.hparams.model_type == 'deri_ref':
    #         # NOTE: your original code uses -q here, but q is local to sub_forward.
    #         # If you really need deri_ref, you must return q from sub_forward, or recompute.
    #         raise NotImplementedError("deri_ref needs q available in forward; return q from sub_forward or remove this mode.")
    #     elif self.hparams.model_type == 'inte':
    #         raise NotImplementedError("inte mode uses state_seq/q not in this forward scope; keep your old logic or refactor.")
    #     elif self.hparams.model_type == 'direct':
    #         out = x
    #     else:
    #         raise NotImplementedError

    #     out = out.view(b, t, -1)
    #     return out, rnn_state


    def get_initial_state(self, batch_size):
        state_size = (1, batch_size, self._lstm[0].hidden_size) # num_layer x bsize x hidden size
        states = []
        for i in range(self.hparams.abnet_heads):
            h_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
            c_state = torch.zeros(state_size, dtype=torch.float32).to(self.device)
            states.append([h_state, c_state])
        return states

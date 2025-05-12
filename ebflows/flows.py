import numpy as np
import torch
from normflows.flows import Flow
from .nets import MLP
from .transforms import Preprocessing
from .distributions import ConditionalDiagLinearGaussian

class MaskedCondAffineFlow(Flow):
    '''RealNVP as introduced in [arXiv: 1605.08803](https://arxiv.org/abs/1605.08803)

    Masked affine flow:

    ```
    f(z) = b * z + (1 - b) * (z * exp(s(b * z)) + t)
    ```

    - class AffineHalfFlow(Flow): is MaskedAffineFlow with alternating bit mask
    - NICE is AffineFlow with only shifts (volume preserving)
    '''

    def __init__(self, b, t=None, s=None):
        '''Constructor

        Args:
          b: mask for features, i.e. tensor of same size as latent data point filled with 0s and 1s
          t: translation mapping, i.e. neural network, where first input dimension is batch dim, if None no translation is applied
          s: scale mapping, i.e. neural network, where first input dimension is batch dim, if None no scale is applied
        '''
        super().__init__()
        self.b_cpu = b.view(1, *b.size())
        self.register_buffer('b', self.b_cpu)
        self.s = s
        self.t = t

    def get_st(self, z_masked, context):
        tmp = torch.cat([z_masked, context], dim=1)
        scale = self.s(tmp) if self.s is not None else torch.zeros_like(z_masked)
        trans = self.t(tmp) if self.t is not None else torch.zeros_like(z_masked)
        return scale, trans

    def forward(self, z, context):
        z_masked = self.b * z
        scale, trans = self.get_st(z_masked, context)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def inverse(self, z, context):
        z_masked = self.b * z
        scale, trans = self.get_st(z_masked, context)
        z_ = z_masked + (1 - self.b) * (z - trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det

    def get_qv(self, z, context):
        z_, log_det = self.inverse(z, context)
        q = log_det
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v

class CondScaling(Flow):
    '''
    Transformation: z_ = z * exp(s) / exp(s)
    '''
    def __init__(self, s1, s2=None):
        super().__init__()
        self.scale1 = s1
        self.scale2 = s2

    def forward(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det

    def inverse(self, z, context):
        log_det = torch.zeros(z.shape[0], device=z.device)
        return z, log_det

    def get_qv(self, z, context):
        if self.scale2 is not None:
            s1 = self.scale1(context[:context.shape[0]//2])
            s2 = self.scale2(context[:context.shape[0]//2])
            q = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
            v = torch.cat((s1[:, 0], s2[:, 0]), dim=0)
        else:
            s1 = self.scale1(context)
            q = s1[:, 0]
            v = s1[:, 0]
        return z, q, v

def initializeFlow(log_sigma_max, log_sigma_min, action_sizes, state_sizes):
    '''
    We reuse this function from the original MEow paper, due to strict
    initialization requirements for the flow policy.
    '''

    dropout_rate_flow = 0.1
    dropout_rate_scale = 0.0
    layer_norm_flow = True
    layer_norm_scale = False
    hidden_layers = 2
    flow_layers = 2
    hidden_sizes = 64
    scale_hidden_sizes = 256

    # Construct the prior distribution and the linear transformation
    prior_list = [state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
    log_scale = MLP(prior_list, init='zero')
    q0 = ConditionalDiagLinearGaussian(action_sizes, log_scale, LOG_SIGMA_MIN=log_sigma_min, LOG_SIGMA_MAX=log_sigma_max)

    # Construct normalizing flow
    flows = []
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(action_sizes)])
    for i in range(flow_layers):
        layers_list = [action_sizes+state_sizes] + [hidden_sizes]*hidden_layers + [action_sizes]
        s = None
        t1 = MLP(layers_list, init='orthogonal', dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        t2 = MLP(layers_list, init='orthogonal', dropout_rate=dropout_rate_flow, layernorm=layer_norm_flow)
        flows += [MaskedCondAffineFlow(b, t1, s)]
        flows += [MaskedCondAffineFlow(1 - b, t2, s)]

    # Construct the reward shifting function
    scale_list = [state_sizes] + [scale_hidden_sizes]*hidden_layers + [1]
    learnable_scale_1 = MLP(scale_list, init='zero', dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    learnable_scale_2 = MLP(scale_list, init='zero', dropout_rate=dropout_rate_scale, layernorm=layer_norm_scale)
    flows += [CondScaling(learnable_scale_1, learnable_scale_2)]

    # Construct the preprocessing layer
    flows += [Preprocessing()]
    return flows, q0
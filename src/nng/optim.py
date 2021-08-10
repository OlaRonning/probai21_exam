import torch
from torch.optim import Optimizer


def _noisy_adam(params, grads, beta1, beta2, lr, kl_weight, prior_var, ext_damping):
    pass


class NoisyAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(.9, .999), kl_weight=1., prior_var=.5, ext_damping=0.):
        defaults = dict(lr=lr, betas=betas, kl_weight=kl_weight, prior_var=prior_var, ext_damping=ext_damping)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure

        for group in self.param_groups:
            params_with_grad = []
            grads = []

            beta1, beta2 = group['betas']
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('NoisyAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        # TODO: instantiate state

                    state['step'] += 1
                    state_steps.append(state['step'])

            _noisy_adam(params_with_grad,
                        grads,
                        beta1=beta1,
                        beta2=beta2,
                        lr=group['lr'],
                        kl_weight=group['kl_weight'],
                        prior_var=group['prior_var'],
                        ext_damping=group['ext_damping'])

        return loss

import torch

from torch.optim import Optimizer


class Optimizer_1(Optimizer):
    """Implements Optimizer_1 algorithm.
    It was been proposed in `http://proceedings.mlr.press/v70/bello17a/bello17a.pdf`.
    Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining parameter groups
      lr (float, optional): learning rate (default: 1)
      beta (float, optional): coefficients used for computing running averages of gradient
          (default: 0.9)
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1, beta=0.9, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
      Arguments:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
      """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()

                exp_avg = state['exp_avg']
                beta = group['beta']

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Original Variable Ref: https://arxiv.org/abs/1412.6980
                # g_t = grad
                # B = beta
                # m_{t - 1} = exp_avg
                # t = step

                # Neural Optimizer Search Optimizer_1 Ref:
                # http://proceedings.mlr.press/v70/bello17a/bello17a.pdf
                # update = g_t * e^{\sign{g} * sign{m_t}}

                # exp_avg = exp_avg * beta + (1 - beta) * grad
                # m_t = m_{t - 1} * B + (1 - B) * g_t
                exp_avg.mul_(beta).add_(1 - beta, grad)
                # update = g_t * e^(\sign{g_t} * \sign{m_t})
                update = grad.mul(torch.exp(torch.sign(grad) * torch.sign(exp_avg)))
                # paramer = parameter - lr * update
                p.data.add_(-group['lr'], update)

        return loss

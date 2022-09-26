from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta1, beta2 = torch.tensor(beta1), torch.tensor(beta2)
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # todo

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                # Hermes - see pytorch implementation for further hints
                # State initialization
                if not state or len(state) == 0:

                    # Exponential moving average of gradient values
                    state['moment_one'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['moment_two'] = torch.zeros_like(p)
                    state["time"] = 0

                else:
                    state["moment_one"] = (state["moment_one"] * beta1) + (1-beta1) * grad
                    state["moment_two"] = (state["moment_two"] * beta2) + (1 - beta2) * grad ** 2


                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                state["time"] += 1
                t = state["time"]
                alpha_t = (alpha * math.sqrt(1 - beta2 ** t)) / (1 - beta1 ** t)

                # Update parameters

                update = state["moment_one"] / (torch.sqrt(state["moment_two"]) + eps)
                p.data -= ((alpha_t * update) + (alpha * group["weight_decay"] * p.data))

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                #p.data -= (alpha * group["weight_decay"] * p.data)
                #p.data.mul_(1 - alpha * group["weight_decay"])

        return loss

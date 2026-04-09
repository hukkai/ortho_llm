from __future__ import annotations

import torch
import torch.distributed as dist

from .ops import polar #, fast_exp_action
#from .fuse_ops import update_fused
from .polar_taylor import stiefel_update_taylor

class SOOptimizer:
    def __init__(
        self,
        param: torch.nn.Parameter,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        sub_matrix: int = 8,
        project_last: bool = True,
    ) -> None:
        self.param = param
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.project_last = project_last

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        total = param.shape[0]
        if total % self.world_size != 0:
            raise ValueError("chunk_weights must be divisible by world size")

        per_rank = total // self.world_size
        self.local_slice = slice(self.rank * per_rank, (self.rank + 1) * per_rank)

        self.m = torch.zeros_like(param.data[self.local_slice])
        self.v = torch.zeros_like(self.m)
        self.buffer = torch.zeros_like(param.data)
        self.step_count = torch.tensor(0.0, device=self.m.device)

        self.dim = self.m.shape[1]
        if self.dim % sub_matrix != 0:
            raise ValueError(f"Matrix dim {self.dim} must be divisible by sub_matrix {sub_matrix}")

        self.orth_dim = self.dim // sub_matrix

    def state_dict(self) -> dict:
        return {
            "m": self.m,
            "v": self.v,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "project_last": self.project_last,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.m = state.get("m", self.m).to(device=self.m.device, dtype=self.m.dtype)
        self.v = state.get("v", self.v).to(device=self.v.device, dtype=self.v.dtype)
        self.lr = state.get("lr", self.lr)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.eps = state.get("eps", self.eps)
        self.project_last = state.get("project_last", self.project_last)
        self.step_count = state.get("step_count", self.step_count).to(
            device=self.step_count.device, dtype=self.step_count.dtype
        )

    def step(self, lr: float | None = None, is_last: bool = False) -> None:
        if self.param.grad is None:
            return

        lr = lr if lr is not None else self.lr
        self.step_count += 1

        x = self.param.data[self.local_slice]
        grad = self.param.grad[self.local_slice]

        self.m += (grad - self.m) * (1.0 - self.beta1)
        self.v += (grad.pow(2) - self.v) * (1.0 - self.beta2)

        m_hat = self.m / (1.0 - self.beta1**self.step_count)
        v_hat = self.v / (1.0 - self.beta2**self.step_count)
        update = -m_hat / (v_hat.sqrt() + self.eps) * lr

        x = x.reshape(-1, self.orth_dim, self.dim)
        update = update.reshape_as(x)
        # new_x = fast_exp_action(x, update)
        # new_x = update_fused(x, update)
        new_x = stiefel_update_taylor(x, update)

        if is_last and self.project_last:
            new_x = polar(new_x)

        new_x = new_x.reshape_as(self.m)

        self.buffer.zero_()
        self.buffer[self.local_slice] = new_x
        if dist.is_initialized():
            dist.all_reduce(self.buffer)
        self.param.data.copy_(self.buffer)
        self.param.grad = None

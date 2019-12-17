import torch
import numpy as np


def positional_encoding(n_positions: int, hidden_dim: int) -> torch.Tensor:
    def calc_angles(pos, i):
        rates = 1 / np.power(10000, (2*(i // 2)) / np.float32(hidden_dim))
        return pos * rates

    rads = calc_angles(np.arange(n_positions)[:, np.newaxis], np.arange(hidden_dim)[np.newaxis, :])

    rads[:, 0::2] = np.sin(rads[:, 0::2])
    rads[:, 1::2] = np.cos(rads[:, 1::2])

    pos_enc = rads[np.newaxis, ...]
    pos_enc = torch.tensor(pos_enc, dtype=torch.float32, requires_grad=False)
    return pos_enc


def dense_interpolation(batch_size: int, seq_len: int, factor: int) -> torch.Tensor:
    W = np.zeros((factor, seq_len), dtype=np.float32)
    for t in range(seq_len):
        s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
        for m in range(factor):
            tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
            w = np.power(tmp, 2, dtype=np.float32)
            W[m, t] = w

    W = torch.tensor(W, requires_grad=False).float().unsqueeze(0)
    return W.repeat(batch_size, 1, 1)


def subsequent_mask(size: int) -> torch.Tensor:
    """
    from Harvard NLP
    The Annotated Transformer

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking

    :param size: int
    :return: torch.Tensor
    """
    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("float32")
    mask = torch.from_numpy(mask) == 0
    return mask.float()


class ScheduledOptimizer:
    """
    Reference: `jadore801120/attention-is-all-you-need-pytorch \
    <https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py>`_
    """
    def __init__(self, optimizer, d_model: int, warm_up: int) -> None:
        self._optimizer = optimizer
        self.warm_up = warm_up
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> np.array:
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.warm_up, -1.5) * self.n_current_steps
        ])

    def get_lr(self):
        lr = self.init_lr * self._get_lr_scale()
        return lr

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return self._optimizer.state_dict()

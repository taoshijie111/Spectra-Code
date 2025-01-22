import torch


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        self.power = kwargs.get('power')
        self.max_step = kwargs.get('max_step')
        self.warmup_step = kwargs.get('warmup_step')
        self.warmup_factor = kwargs.get('warmup_factor', 1 / 4)
        self.warmup_method = kwargs.get('warmup_method', 'linear')
        self.target_lr = kwargs.get('target_lr', 1e-8)

        if self.warmup_method not in ('constant', 'linear'):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted "
                'got {}'.format(self.warmup_method))

        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _get_warmup_factor_at_iter(method: str, iter: int, warmup_steps: int,
                                   warmup_factor: float) -> float:
        """
        Return the learning rate warmup factor at a specific iteration.
        See :paper:`in1k1h` for more details.

        Args:
            method (str): warmup method; either "constant" or "linear".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_steps (int): the number of warmup iterations.
            warmup_factor (float): the base warmup factor (the meaning changes according
                to the method used).

        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= warmup_steps:
            return 1.0

        if method == 'constant':
            return warmup_factor
        elif method == 'linear':
            alpha = iter / warmup_steps
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError('Unknown warmup method: {}'.format(method))

    def get_lr(self):
        """
        获取当前的学习率

        Returns:
            list[float]: 学习率
        """
        N = self.max_step - self.warmup_step
        T = self.last_epoch - self.warmup_step
        if self.last_epoch <= self.warmup_step:
            warmup_factor = self._get_warmup_factor_at_iter(self.warmup_method,
                                                            self.last_epoch,
                                                            self.warmup_step,
                                                            self.warmup_factor)
            return [
                self.target_lr + (base_lr - self.target_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        factor = pow(1 - T / N, self.power)
        return [
            self.target_lr + (base_lr - self.target_lr) * factor
            for base_lr in self.base_lrs
        ]


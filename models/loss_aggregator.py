import torch


class LossAggregator(object):
    def __init__(self, sub_modules: list["LossAggregator"] = None) -> None:
        self._losses = {}
        self._weights = {}
        self._total_loss = None

        self.sub_modules: list["LossAggregator"] = sub_modules or []

    def add_loss(self, loss_value: torch.Tensor, name: str, loss_weight: torch.Tensor | float = 1.0) -> None:
        if name in self:
            raise ValueError("Aggregator was not properly reset, found duplicate for `{}`".format(name))

        if name == "loss":
            raise ValueError("Name `loss` is reserved for the total loss.")

        self._total_loss = None
        self._losses[name] = loss_value.mean()
        self._weights[name] = loss_weight

    def get_total_loss(self) -> torch.Tensor:
        if self._total_loss is None:
            self._total_loss = sum([value * weight
                                    for _, value, weight in self
                                    if weight != 0.0],
                                   start=0.0)
        return self._total_loss

    def get_losses(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value, _ in self}

    def get_logged_losses(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = self.get_total_loss()
        losses = {key: value.detach() for key, value in self.get_losses().items()}
        losses["loss"] = loss.detach()
        return loss, losses

    def clear(self):
        for sub_module in self.sub_modules:
            sub_module.clear()

        self._losses = {}
        self._weights = {}
        self._total_loss = None

    def __contains__(self, name: str):
        for sub_module in self.sub_modules:
            if name in sub_module:
                return True

        return name in self._losses

    def __iter__(self):
        for sub_module in self.sub_modules:
            for name, loss, weight in sub_module:
                yield name, loss, weight

        for name in self._losses:
            yield name, self._losses[name], self._weights[name]

import torch
import numpy as np


def get_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if len(logits.shape) > 1:
        predicted = logits.argmax(dim=-1)
    else:
        predicted = logits > 0.5
    accuracy = torch.eq(predicted, labels).float().mean()
    return accuracy


def batched_cov(x: torch.Tensor) -> torch.Tensor:
    feature_count = x.shape[2]
    x = x - x.mean(dim=2, keepdim=True)
    covariance = (x @ x.transpose(dim0=1, dim1=2)) / (feature_count - 1)
    return covariance


def batched_corrcoef(*x: torch.Tensor, absolute=True) -> torch.Tensor:
    if len(x) == 1:
        x = x[0]
    elif len(x[0].shape) == 2:
        x = torch.stack(x, dim=1)
    else:
        x = torch.concat(x, dim=1)

    covariance = batched_cov(x)
    cov_diagonal = torch.diagonal(covariance, dim1=1, dim2=2)
    stddev = cov_diagonal.sqrt()
    covariance /= (stddev.unsqueeze(2) * stddev.unsqueeze(1))

    if absolute:
        covariance = torch.abs(covariance)

    return covariance


def format_metric(data: tuple[float, float] | float, std: float | None = None) -> str:
    if std is None:
        mean, std = data
    else:
        mean, std = data, std

    if np.isnan(mean) or np.isnan(std):
        return ""
    return "{} (+/- {})".format(round(mean * 100, 1), round(std * 100, 1))

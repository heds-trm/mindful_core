import numpy as np
import scipy.stats
from statsmodels.stats.multitest import multipletests
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


# region Original 1-1 delong test
# ROC comparison adapted from
# https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py


def delong_roc_variance(ground_truth: np.ndarray,
                        predictions: np.ndarray):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: array of 0 and 1
       predictions: array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delong_cov


def delong_roc_test(ground_truth: np.ndarray,
                    predictions_one: np.ndarray,
                    predictions_two: np.ndarray) -> float:
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: array of 0 and 1
       predictions_one: predictions of the first model,
          array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delong_cov = fast_delong(predictions_sorted_transposed, label_1_count)
    return compute_pvalue(aucs, delong_cov)


def fast_delong(predictions_sorted_transposed: np.ndarray,
                label_1_count: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    The fast version of DeLong's method for computing the covariance of unadjusted AUC.

    Args:
        predictions_sorted_transposed: a 2D array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
        label_1_count: number of positive samples

    Returns:
        (AUC value, DeLong covariance)
    Reference:
        @article{sun2014fast,
                 title={Fast Implementation of DeLong's Algorithm for
                            Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
                 author={Xu Sun and Weichao Xu},
                 journal={IEEE Signal Processing Letters},
                 volume={21},
                 number={11},
                 pages={1389--1393},
                 year={2014},
                 publisher={IEEE}
        }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_mid_rank(positive_examples[r, :])
        ty[r, :] = compute_mid_rank(negative_examples[r, :])
        tz[r, :] = compute_mid_rank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def compute_mid_rank(x: np.ndarray) -> np.ndarray:
    """Computes mid-ranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of mid-ranks
    """
    indices = np.argsort(x)
    sorted_x = x[indices]
    length = len(x)
    raw_mid_ranks = np.zeros(length, dtype=np.float32)

    i = 0
    while i < length:
        j = i
        while (j < length) and (sorted_x[j] == sorted_x[i]):
            j += 1
        raw_mid_ranks[i:j] = i + j - 1
        i = j

    mid_ranks = np.empty(length, dtype=np.float32)
    mid_ranks[indices] = raw_mid_ranks * 0.5 + 1
    return mid_ranks


def compute_pvalue(aucs: np.ndarray, sigma: np.ndarray) -> float:
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(p-value)
    """
    tmp = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(tmp, sigma), tmp.T))
    # noinspection PyUnresolvedReferences
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth: np.ndarray):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


# endregion


def compute_kfolds_delong_roc_test(ground_truth: list[np.ndarray],
                                   predictions_0: list[np.ndarray],
                                   predictions_1: list[np.ndarray],
                                   alpha=5e-2
                                   ) -> tuple[tuple[np.ndarray, np.ndarray],
                                              tuple[np.ndarray, np.ndarray]]:
    log_p_values = [delong_roc_test(fold_ground_truth, fold_predictions_0, fold_predictions_1)
                    for (fold_ground_truth, fold_predictions_0, fold_predictions_1)
                    in zip(ground_truth, predictions_0, predictions_1)]
    p_values = [np.exp(np.squeeze(log_p_value)) for log_p_value in log_p_values]
    p_values = np.asarray(p_values)
    reject = p_values < alpha
    reject_corrected, p_values_corrected, *_ = multipletests(p_values, alpha, method="holm")
    return (reject, p_values), (reject_corrected, p_values_corrected)


def compute_roc_curve(ground_truth: np.ndarray | list[np.ndarray],
                      predictions: np.ndarray | list[np.ndarray],
                      n_thresholds: int = 1000
                      ) -> tuple[np.ndarray, np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
    if isinstance(predictions, list):
        return [compute_roc_curve(fold_ground_truth, fold_predictions, n_thresholds)
                for fold_ground_truth, fold_predictions
                in zip(ground_truth, predictions)]

    ground_truth = ground_truth.astype(np.int32)

    false_positive_rates, true_positive_rates, thresholds = roc_curve(ground_truth, predictions)
    true_positive_rates = resample_curve(thresholds, true_positive_rates, n_thresholds)
    false_positive_rates = resample_curve(thresholds, false_positive_rates, n_thresholds)

    return false_positive_rates, true_positive_rates


def resample_curve(x_values: np.ndarray, y_values: np.ndarray, n_points: int) -> np.ndarray:
    interpolator = interp1d(x_values, y_values, kind="nearest")
    x_min = x_values.min(where=np.isfinite(x_values), initial=1.0)
    x_max = x_values.max(where=np.isfinite(x_values), initial=0.0)
    new_x_values = np.linspace(x_min, x_max, num=n_points)
    return interpolator(new_x_values)


def compute_kfolds_roc_distribution(roc_curves: list[tuple[np.ndarray, np.ndarray]]
                                    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    # noinspection PyTypeChecker
    false_positive_rates, true_positive_rates = np.moveaxis(roc_curves, source=1, destination=0)

    fpr_mean = false_positive_rates.mean(axis=0)
    tpr_mean = true_positive_rates.mean(axis=0)

    fpr_std = false_positive_rates.std(axis=0)
    tpr_std = true_positive_rates.std(axis=0)

    return (fpr_mean, tpr_mean), (fpr_std, tpr_std)

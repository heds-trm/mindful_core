# Experiment Configuration

Experiments are defined through JSON configuration files. A configuration specifies:

- Available datasets
- Shared training settings
- Experiment definitions
- Logging locations

This approach allows experiments to be reproduced and modified without changing the source code.

> [!CAUTION]
> According to JSON specification, you cannot add comments and trailing commas in a .json file leading to possible crashes.
>
> Ex. {"one":1,"two":2,} is forbidden.

## Configuration Overview

### Datasets

The `datasets` section defines the datasets available to experiments. Each dataset is uniquely identified by a dictionary entry, here "mura_dataset".

```json
{
    "datasets": {
        "mura_dataset": {
            "folds": "<folds_dir>"
        }
    }
}
```

| Parameter | Description |
|------------|-------------|
| `folds` | Directory containing the cross-validation folds CSV files for the dataset. |

---

### Shared Settings

The `shared.default` section contains settings applied to all experiments unless explicitly overridden.

```json
{
    "shared": {
        "default": {
            "log_monitors": ["validation_loss_epoch"],
            "save_last": true,
            "batch_size": 32,
            "num_workers": 0,
            "accelerator": "gpu",
            "max_epochs": 10,
            "devices": 1,
            "seeds": [...]
        }
    }
}
```

| Parameter | Description |
|------------|-------------|
| `log_monitors` | Metrics monitored during training. |
| `save_last` | Save the final model checkpoint. |
| `batch_size` | Number of samples per batch. |
| `num_workers` | Number of data loading workers. |
| `accelerator` | Training accelerator (`cpu`, `gpu`, etc.). |
| `max_epochs` | Maximum number of training epochs. |
| `devices` | Number of devices used for training. |
| `seeds` | Random seeds used for repeated runs and reproducibility. One seed per fold, defined as a list, e.g., [12, 434]. |

---

### Experiments

Experiments are uniquely defined by a name, here "unimodal_densenet121_mura":

```json
{
    "unimodal_densenet121_mura": {
        "skip": "no",
        "model": {
            "class_name": "densenet121",
            "hparams": "<models_dir>/densenet121_hparams.json"
        },
        "pipeline_config": "<pipelines_dir>/unimodal_mura.json",
        "dataset": "mura_dataset",
        "folds": "all",
        "stages": "train test"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `skip` | Whether the experiment should be skipped. |
| `dataset` | Dataset identifier defined in the `datasets` section. |
| `folds` | Folds to evaluate provided as a 0-based integer, a list of integers of `all` that includes all available folds. |
| `model.class_name` | Model architecture to instantiate. |
| `model.hparams` | Model hyperparameter [configuration file](./MURA_HPARAMS_CONFIGURATION.md). |
| `pipeline_config` | Data processing and training [pipeline definition](./MURA_PIPELINE_CONFIGURATION.md). |
| `stages` | Stages to execute (`train`, `test`). |

### Logging

```json
{
    "log_dir": "<logs_dir>"
}
```

| Parameter | Description |
|------------|-------------|
| `log_dir` | Directory where logs, checkpoints, metrics, and experiment artifacts are stored. |


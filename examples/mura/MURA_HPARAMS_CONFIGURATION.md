# Hyperparameter Configuration

This configuration defines a DenseNet121-based image classification model for binary classification. It specifies the network architecture, training behavior, and optimizer settings. This configuration file is used in the [MURA example](./MURA_README.md).

## Model Configuration

```json
{
    "model_name": "densenet121",
    "spatial_dims": 2,
    "feed_forward": true,
    "class_count": 2,
    "yield_confidence": false,
    "label_smoothing": 0
}
```

| Parameter | Description |
|------------|-------------|
| `model_name` | Model architecture to instantiate. |
| `spatial_dims` | Number of spatial dimensions in the input data. A value of `2` indicates 2D images. |
| `feed_forward` | Indicates that the model performs standard feed-forward inference. |
| `class_count` | Number of output classes. |
| `yield_confidence` | Whether confidence estimates are returned alongside predictions. |
| `label_smoothing` | Amount of label smoothing applied during training. A value of `0` disables label smoothing. |

<details>
<summary>Which models are currently supported?</summary>

  look at `SUPPORTED_BACKBONES` list in [models/classification/monai_classifier.py](../../models/classification/monai_classifier.py), which are based on MONAI models.
</details>

---

## Backbone Configuration

The backbone configuration defines architecture-specific settings, these parameters are passed to the corresponding MONAI model constructor.

```json
{
    "backbone_config": {
        "dropout_prob": 0.0,
        "pretrained": true
    }
}
```

| Parameter | Description |
|------------|-------------|
| `dropout_prob` | Dropout probability applied within the network, refer to corresponding MONAI documentation. |
| `pretrained` | Whether pretrained weights are used to initialize the model. Often this is only available for 2D image models. |

> [!NOTE]
> 
> - `pretrained: true` initializes DenseNet121 using pretrained weights.
> - Transfer learning often improves convergence speed and performance when training datasets are limited.

---

## Optimizer Configuration

The optimizer configuration specifies how model parameters are updated during training.

```json
{
    "optimizer_config": {
        "optimizer": {
            "optimizer_type": "adam",
            "lr": 0.0001
        }
    }
}
```

| Parameter | Description |
|------------|-------------|
| `optimizer_type` | Optimization algorithm used during training. |
| `lr` | Learning rate. |

<details>
<summary>Which optimizers are currently supported?</summary>

  look at `make_optimizer` function in [models/module.py](../../models/module.py), which are based on MONAI optimizers.
</details>

> [!NOTE]
> 
> - The Adam optimizer is used for gradient-based optimization.
> - The learning rate is set to **1e-4**, a commonly used value for fine-tuning pretrained convolutional neural networks.

---

## Configuration Summary

This configuration defines:

- A **DenseNet121** image classification model.
- **2D image inputs**.
- **Binary classification** (`class_count = 2`).
- **Pretrained ImageNet weights**.
- **No label smoothing**.
- **No dropout regularization**.
- **Adam optimization** with a learning rate of **0.0001**.


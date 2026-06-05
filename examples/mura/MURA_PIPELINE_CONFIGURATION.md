# Pipeline Configuration

A pipeline defines the preprocessing and data augmentation operations applied to dataset modalities before training and evaluation. The pipeline is organized into stages, where each stage consists of a sequence of transformations applied to one or more modalities. Most operations are serialized versions of MONAI transforms.

We present the example of the pipeline applied in the [MURA example](./MURA_README.md).

## Inputs and Outputs

### Inputs

| Name | Description |
|--------|-------------|
| `image` | A medical image. |
| `label` | A Ground-truth label associated with the image. |

```json
{
    "inputs": {
        "image": "image",
        "label": "label"
    }
}
```

> [!NOTE]
> These names must match the column headers defined in the folds CSV files.

### Outputs

Same goes for the data out of the pipeline:

```json
{
    "outputs": {
        "image": "image",
        "label": "label"
    }
}
```

---

# Preprocessing Stage

The `preprocess` stage is executed before training and inference. Its purpose is to load the data, normalize its format, and prepare it for the model.

## Processing Sequence

### 1. Load Image

Loads the image from disk.

```json
{
    "type": "load_image",
    "parameters": {
        "image_only": true,
        "ensure_channel_first": true
    }
}
```

| Parameter | Description |
|------------|-------------|
| `image_only` | Load only the image data. |
| `ensure_channel_first` | Ensures the channel dimension is the first dimension. |

---

### 2. Convert to Grayscale

Converts the image to grayscale.

```json
{
    "type": "to_grayscale",
    "parameters": {
        "channel_first": true
    }
}
```

| Parameter | Description |
|------------|-------------|
| `channel_first` | Indicates that channels are stored in the first dimension. |

---

### 3. Resize

Resizes images to a fixed spatial resolution.

```json
{
    "type": "resize",
    "parameters": {
        "spatial_size": [512, 512],
        "mode": "trilinear",
        "anti_aliasing": false
    }
}
```

| Parameter | Description |
|------------|-------------|
| `spatial_size` | Target image size. |
| `mode` | Interpolation mode used during resizing. |
| `anti_aliasing` | Enables or disables anti-aliasing. |

---

### 4. Intensity Standardization

Normalizes image intensities.

```json
{
    "type": "standardize_intensity",
    "parameters": {
        "channel_wise": false,
        "spatial_dims": 2
    }
}
```

| Parameter | Description |
|------------|-------------|
| `channel_wise` | Whether normalization is performed independently for each channel. |
| `spatial_dims` | Number of spatial dimensions in the image. |

---

### 5. Convert Image Type

Converts the image to the desired tensor type.

```json
{
    "type": "ensure_type",
    "parameters": {
        "device": "cpu",
        "dtype": "float32"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `device` | Device on which the tensor is created. |
| `dtype` | Target data type. |

---

### 6. Convert Label to Tensor

Converts labels into tensors.

```json
{
    "type": "to_tensor",
    "parameters": {
        "dtype": "float32",
        "device": "cpu"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `dtype` | Target data type. |
| `device` | Device on which the tensor is created. |

---

# Data Augmentation Stage

The `view_augment` stage is applied during training to improve model robustness and reduce overfitting. 

<details>
<summary>Which transforms are serializable?</summary>

  look at [data/transforms/serializable_transform.py](../../data/transforms/serializable_transform.py)
</details>

## Transformations

### Random Flip

Randomly flips the image.

```json
{
    "type": "rand_flip",
    "parameters": {
        "prob": 0.5
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the transformation. |

---

### Random Rotation

Randomly rotates the image.

```json
{
    "type": "rand_rotate",
    "parameters": {
        "prob": 0.8,
        "range_x": 0.34,
        "range_y": 0.34,
        "padding_mode": "zeros"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the transformation. |
| `range_x` | Maximum rotation range around the x-axis in radians. |
| `range_y` | Maximum rotation range around the y-axis in radians. |
| `padding_mode` | Strategy used to fill empty pixels after rotation. |

---

### Random Affine Transformation

Applies random translations to the image.

```json
{
    "type": "rand_affine",
    "parameters": {
        "prob": 0.5,
        "translate_range": [2, 2],
        "padding_mode": "border"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the transformation. |
| `translate_range` | Maximum translation in pixels. |
| `padding_mode` | Padding strategy for empty regions. |

---

### Salt-and-Pepper Noise

Applies random impulse noise.

```json
{
    "type": "rand_salt_pepper_noise",
    "parameters": {
        "prob": 0.8,
        "ratio": 0.1
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the transformation. |
| `ratio` | Fraction of pixels affected by the noise. |

---

### Gaussian Noise

Adds Gaussian noise to the image.

```json
{
    "type": "rand_gaussian_noise",
    "parameters": {
        "prob": 0.2,
        "std": 0.01
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the transformation. |
| `std` | Standard deviation of the Gaussian noise. |

---

# Pipeline Summary

This pipeline performs the following operations:

1. Load image data.
2. Convert images to grayscale.
3. Resize all images to **512 × 512**.
4. Standardize image intensities.
5. Convert images and labels to tensors.
6. Apply data augmentation during training:
   - Random flipping
   - Random rotations
   - Random affine translations
   - Salt-and-pepper noise
   - Gaussian noise

This preprocessing ensures that all samples have a consistent format while the augmentation stage improves generalization by exposing the model to realistic variations of the input data.


# GradICONLung

`GradICONLung` is a deep-learning registration model for lung CT. It uses the [uniGradICON](https://github.com/uncbiag/uniGradICON) network, trained with [GradICON](https://github.com/uncbiag/ICON) on lung CT only. Unlike the GradICON lung model in `icon_registration`, it takes unmasked images. It is trained on inspiration-expiration CT pairs from COPDGene, with the DirLab patients held out, using the GradICON regularizer with weight 4.5 and LNCC with sigma 1 inside the lung masks.

It can also return a stationary velocity field (SVF) v, so that the transform is exp(v) and its inverse is exp(-v).

## Installation

```
python3 -m venv gradiconlung_virtualenv
source gradiconlung_virtualenv/bin/activate

git clone https://github.com/uncbiag/GradICONLung.git
cd GradICONLung
pip install .
```

This installs `gradiconlung-register` and `gradiconlung-warp`. The weights are downloaded into `~/.cache/gradiconlung/` on first use, or you can pass them with `--network_weights PATH`.

## Input images

Both images must be CT in HU. They are clipped to [-1000, 1000] HU and mapped to [0, 2] by (HU + 1000) / 1000, as in training. Lung segmentations are not needed.

## Register a pair of images

```
gradiconlung-register --fixed fixed.nii.gz --moving moving.nii.gz \
    --transform_out trans.hdf5 --warped_moving_out warped_moving.nii.gz
```

The transform maps fixed image points to the moving image. By default, the result is refined with 50 steps of instance optimization (IO). Add `--io_iterations None` to skip IO.

### SVF output

```
gradiconlung-register --fixed fixed.nii.gz --moving moving.nii.gz \
    --transform_out trans.hdf5 --warped_moving_out warped_moving.nii.gz --svf_out svf.nii.gz
```

An SVF v is fitted so that exp(v) matches the forward transform and exp(-v) the backward transform. With `--svf_out`, the transform and the warped image are computed from exp(v), and `--transform_out` becomes optional. The SVF is a 3-component vector image in mm on the 175×175×175 network grid over the fixed image, in ITK physical coordinates.

| Option        | Description                             | Default |
| ------------- | --------------------------------------- | ------- |
| `--svf_out`   | Path of the SVF. Turns on the SVF fit   | off     |
| `--svf_steps` | Maximum number of steps of the SVF fit  | 50      |
| `--svf_lr`    | Learning rate of the SVF fit            | 0.5     |

### Instance optimization

The IO uses the training loss, the GradICON regularizer with weight 4.5 and LNCC with sigma 1.

| Option            | Description                                    | Default   |
| ----------------- | ---------------------------------------------- | --------- |
| `--io_iterations` | Number of IO steps, or `None` to skip IO       | 50        |
| `--io_sim`        | Similarity measure, `lncc`, `lncc2` or `mind`  | `lncc`    |
| `--sigma`         | Gaussian kernel size of `lncc` and `lncc2`     | 1         |
| `--io_lr`         | Learning rate                                  | 0.00002   |

More steps are slower but more accurate. For the best accuracy, use 200 steps with lung masks and `--loss_function_masking`.

### Masks and segmentations

`--loss_function_masking` restricts the IO loss to binary masks of both images, such as lung masks. This matches training and usually improves the result.

```
gradiconlung-register --fixed fixed.nii.gz --fixed_mask fixed_mask.nii.gz \
    --moving moving.nii.gz --moving_mask moving_mask.nii.gz \
    --transform_out trans.hdf5 --io_iterations 200 --loss_function_masking
```

`--input_masking` sets the images to zero outside the masks instead. This is not recommended, because the network was trained on unmasked images.

`--dice_loss_weight` adds a Dice loss between the segmentations of both images to the IO. Use the same label for a structure in both images.

```
gradiconlung-register --fixed fixed.nii.gz --fixed_segmentation fixed_seg.nii.gz \
    --moving moving.nii.gz --moving_segmentation moving_seg.nii.gz \
    --transform_out trans.hdf5 --dice_loss_weight 1.0
```

Masks and segmentations also work with `--svf_out`.

### Intensity conservation

For registrations that must conserve mass, such as inspiration-expiration lung CT, `--intensity_conservation_loss` multiplies the warped moving intensities by the Jacobian determinant of the transform before comparing them with the fixed image. This follows from the change of variables rule. It is only valid for CT, where air (-1000 HU) maps to 0.

```
gradiconlung-register --fixed fixed.nii.gz --moving moving.nii.gz --transform_out trans.hdf5 \
    --intensity_conservation_loss
```

## Warp an image

```
gradiconlung-warp --fixed fixed.nii.gz --moving moving.nii.gz --transform trans.hdf5 \
    --warped_moving_out warped.nii.gz --linear
```

For label maps, use `--nearest_neighbor` instead of `--linear`.

## Comparison on DirLab

Mean landmark error (mTRE, mm) and percentage of folded voxels (%|J|<0) on the 10 DirLab COPDGene cases, with 300 landmarks per case mapped from inspiration to expiration. The initial mTRE is 23.36 mm. All models use the same IO, 50 steps of LNCC with sigma 1 or 5 inside the lung masks, GradICON weight 1.5 and learning rate 0.00002. The compared models are the GradICON lung model of `icon_registration` ([GradICON](https://arxiv.org/abs/2206.05897)), [uniGradICON](https://arxiv.org/abs/2403.05780) and [multiGradICON](https://arxiv.org/abs/2408.00221).

| Model                    | Input       | w/o IO mTRE | w/o IO %\|J\|<0 | IO, LNCC σ=1 mTRE | IO, LNCC σ=1 %\|J\|<0 | IO, LNCC σ=5 mTRE | IO, LNCC σ=5 %\|J\|<0 |
| ------------------------ | ----------- | ----------- | --------------- | ----------------- | --------------------- | ----------------- | --------------------- |
| GradICON lung model      | lung-masked | 1.93        | 2.6e-4          | 1.19              | 9.6e-4                | **1.25**          | 3.7e-4                |
| uniGradICON              | lung-masked | 2.26        | 9.1e-5          | 1.27              | 2.3e-4                | 1.33              | 3.0e-4                |
| multiGradICON            | lung-masked | 2.94        | 1.1e-5          | 1.48              | **3.4e-5**            | 1.53              | 2.4e-5                |
| GradICONLung (ours)      | unmasked    | **1.77**    | 6.2e-5          | **1.15**          | 1.1e-4                | 1.36              | **3.7e-6**            |
| GradICONLung (ours)      | lung-masked | 14.44       | **0**           | 1.63              | 2.0e-4                | 1.51              | 6.2e-5                |

The other models were trained on lung-masked images and need lung segmentations even without IO. GradICONLung needs none, and it is the most accurate without IO and with LNCC sigma 1.

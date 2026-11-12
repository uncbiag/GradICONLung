"""GradICONLung: the uniGradICON network with weights trained for lung CT."""
import hashlib
import os
from pathlib import Path

import itk
import numpy as np
import torch
import torch.nn.functional as F

from icon_registration import config
from icon_registration.itk_wrapper import DEFAULT_FINETUNE_LEARNING_RATE, finetune_execute, register_pair, register_pair_with_mask
import unigradicon
from unigradicon import compute_jacobian_map_command, make_network, make_sim, maybe_cast, warp_command

from . import svf

# As in GradICON's lung training: CT is clipped to [-1000, 1000] HU and mapped to [0, 2] by (HU + 1000) / 1000.
CT_WINDOW = (-1000, 1000)
CT_SCALE = 1000

WEIGHTS_URL = "https://github.com/uncbiag/GradICONLung/releases/download/gradiconlung_weights/Step_2_final.trch"
WEIGHTS_SHA256 = "0ca21936454cb86d28081e9b0dd4424374430627cc82b5c9e689e9f7b44e573e"

# Weight of the GradICON regularizer and LNCC sigma the network was trained with; also used for IO.
LAMBDA = 4.5
SIGMA = 1

SIMILARITIES = ("lncc", "lncc2", "mind")


def default_weights_location():
    """Where the weights are downloaded to: ``$XDG_CACHE_HOME/gradiconlung`` or ``~/.cache/gradiconlung``."""
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return str(cache / "gradiconlung" / "Step_2_final.trch")


def download_weights(weights_location):
    import urllib.request

    print(f"Downloading pretrained GradICONLung weights to {weights_location}")
    os.makedirs(os.path.dirname(weights_location), exist_ok=True)
    partial = weights_location + ".part"
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, partial)
    except Exception as error:
        if os.path.exists(partial):
            os.remove(partial)
        raise RuntimeError(f"Could not download the weights from {WEIGHTS_URL} ({error}). Download them manually "
                           "and pass their path with --network_weights.") from error
    sha256 = hashlib.sha256()
    with open(partial, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    if sha256.hexdigest() != WEIGHTS_SHA256:
        os.remove(partial)
        raise RuntimeError("The downloaded weights are corrupted (checksum mismatch). Please try again, or download "
                           "them manually and pass their path with --network_weights.")
    os.replace(partial, weights_location)


def get_gradiconlung(loss_fn=None, apply_intensity_conservation_loss=False, dice_loss_weight=0.0,
                        loss_function_masking=False, weights_location=None):
    """The uniGradICON network with the lung CT weights. ``loss_fn`` is the IO similarity (default LNCC, sigma 1)."""
    net = make_network(
        unigradicon.input_shape,
        include_last_step=True,
        lmbda=LAMBDA,
        loss_fn=make_sim("lncc", sigma=SIGMA) if loss_fn is None else loss_fn,
        apply_intensity_conservation_loss=apply_intensity_conservation_loss,
        dice_loss_weight=dice_loss_weight,
        loss_function_masking=loss_function_masking,
    )
    if weights_location is None:
        weights_location = default_weights_location()
        if not os.path.exists(weights_location):
            download_weights(weights_location)
    elif not os.path.exists(weights_location):
        raise FileNotFoundError(f"Network weights file not found: {weights_location}")
    print(f"Loading weights from {weights_location}")
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=True)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net


def preprocess(image, mask=None):
    """Clip a CT image (in HU) to [-1000, 1000] and map it to [0, 2], optionally applying a binary mask."""
    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    image = itk.clamp_image_filter(image, Bounds=CT_WINDOW)
    image = itk.shift_scale_image_filter(image, shift=-CT_WINDOW[0], scale=1 / CT_SCALE)
    if mask is not None:
        image = unigradicon.apply_mask(image, mask)
    return image


def _to_network(image, shape, labels=False):
    tensor = torch.tensor(np.asarray(image), dtype=torch.float32, device=config.device)[None, None]
    if labels:
        return F.interpolate(tensor, size=shape, mode="nearest")
    return F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)


def network_maps(model, image_A, image_B, finetune_steps=None, learning_rate=DEFAULT_FINETUNE_LEARNING_RATE,
                 mask_A=None, mask_B=None, segmentation_A=None, segmentation_B=None):
    """Run the network (and IO) and return phi_AB and phi_BA on the network grid."""
    if finetune_steps == 0:
        raise ValueError("To indicate no finetune_steps, pass finetune_steps=None")
    model.to(config.device)
    shape = model.identity_map.shape[2:]
    A, B = _to_network(image_A, shape), _to_network(image_B, shape)
    labels = dict(mask_A=mask_A, mask_B=mask_B, segmentation_A=segmentation_A, segmentation_B=segmentation_B)
    kwargs = {name: _to_network(image, shape, labels=True) for name, image in labels.items() if image is not None}
    if finetune_steps is None:
        with torch.no_grad():
            model(A, B, **kwargs)
    else:
        finetune_execute(model, A, B, finetune_steps, learning_rate, **kwargs)
    with torch.no_grad():
        return model.phi_AB(model.identity_map), model.phi_BA(model.identity_map)


def register_pair_svf(model, image_A, image_B, finetune_steps=None, learning_rate=DEFAULT_FINETUNE_LEARNING_RATE,
                      mask_A=None, mask_B=None, segmentation_A=None, segmentation_B=None,
                      svf_steps=50, svf_lr=0.5, svf_tol=1e-4, exp_steps=7, verbose=True):
    """Register image_A (moving) to image_B (fixed) and fit a stationary velocity field v to the result.

    Returns (phi_AB, phi_BA, velocity): ITK transforms of exp(v) and exp(-v), which replace the network's
    phi_AB and phi_BA, and v as an ITK vector image (mm, ITK physical coordinates) on the network grid
    laid over image_B.
    """
    phi_AB, phi_BA = network_maps(model, image_A, image_B, finetune_steps, learning_rate,
                                  mask_A, mask_B, segmentation_A, segmentation_B)
    if verbose and not svf.same_physical_box(image_A, image_B):
        print("Warning: the moving and fixed images cover different physical boxes. The SVF describes the "
              "deformation after the moving box is mapped onto the fixed box; the ITK transforms include "
              "that mapping.")

    spacing = svf.network_spacing(image_B, model.identity_map.shape[2:])
    disp_AB = svf.displacement_mm(phi_AB, model.identity_map, spacing)
    disp_BA = svf.displacement_mm(phi_BA, model.identity_map, spacing)
    v = svf.fit_svf(disp_AB, disp_BA, spacing, svf_steps, svf_lr, svf_tol, exp_steps)
    with torch.no_grad():
        if verbose:
            rms_AB = torch.sqrt(torch.mean(torch.sum((svf.exp(v, spacing, exp_steps) - disp_AB) ** 2, dim=1)))
            rms_BA = torch.sqrt(torch.mean(torch.sum((svf.exp(-v, spacing, exp_steps) - disp_BA) ** 2, dim=1)))
            print(f"SVF fit: RMS distance to the network's maps is {rms_AB.item():.3f} mm (forward) "
                  f"and {rms_BA.item():.3f} mm (backward)")
        transforms = (
            svf.itk_transform(v, spacing, image_A, image_B, exp_steps),
            svf.itk_transform(-v, spacing, image_B, image_A, exp_steps),
        )
    return transforms + (svf.velocity_image(v, image_B),)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Register two lung CT images using GradICONLung.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image (CT in HU).")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the moving image (CT in HU).")
    parser.add_argument("--fixed_segmentation", required=False,
                         type=str, help="The path of the segmentation of the fixed image, used for Dice loss (see --dice_loss_weight).")
    parser.add_argument("--moving_segmentation", required=False,
                         type=str, help="The path of the segmentation of the moving image, used for Dice loss (see --dice_loss_weight).")
    parser.add_argument("--fixed_mask", required=False,
                         type=str, help="The path of the binary mask for the fixed image.")
    parser.add_argument("--moving_mask", required=False,
                         type=str, help="The path of the binary mask for the moving image.")
    parser.add_argument("--input_masking", required=False,
                         action="store_true", help="Apply the provided masks to input image intensities before registration.")
    parser.add_argument("--loss_function_masking", required=False,
                         action="store_true", help="Apply loss/similarity masking using the provided masks.")
    parser.add_argument("--transform_out", required=False,
                         type=str, help="The path to save the transform.")
    parser.add_argument("--warped_moving_out", required=False,
                        default=None, type=str, help="The path to save the warped image.")
    parser.add_argument("--svf_out", required=False, default=None, type=str,
                         help="The path to save a stationary velocity field (SVF) fitted to the registration. "
                              "If set, --transform_out and --warped_moving_out are computed from exp(SVF).")
    parser.add_argument("--svf_steps", required=False, default=50, type=int,
                         help="The maximum number of steps of the SVF fit. Default is 50.")
    parser.add_argument("--svf_lr", required=False, default=0.5, type=float,
                         help="The learning rate of the SVF fit. Default is 0.5.")
    parser.add_argument("--io_iterations", required=False,
                         default="50", help="The number of IO iterations. Default is 50. Set to 'None' to disable IO.")
    parser.add_argument("--io_lr", required=False, type=float, default=DEFAULT_FINETUNE_LEARNING_RATE,
                         help="The learning rate for instance optimization. Default is 0.00002.")
    parser.add_argument("--io_sim", required=False, default="lncc", choices=SIMILARITIES,
                         help="The similarity measure used in IO. Default is lncc.")
    parser.add_argument("--sigma", required=False, default=SIGMA, type=float,
                         help="Sigma of the Gaussian kernel of lncc and lncc2. Default is 1, as in training.")
    parser.add_argument("--dice_loss_weight", required=False, type=float, default=0.0,
                         help="The weight of the Dice loss if segmentations are provided. Default is 0.0 (no Dice loss).")
    parser.add_argument("--intensity_conservation_loss", required=False,
                         action="store_true", help="Enable determinant-based intensity correction in the loss "
                                                   "function for mass-conserving registration, where -1000 HU represents air.")
    parser.add_argument("--network_weights", required=False, type=str, default=None,
                         help="Path to the network weights. If omitted, they are downloaded on first use.")

    args = parser.parse_args()

    if args.transform_out is None and args.svf_out is None:
        parser.error("Pass --transform_out, --svf_out, or both.")
    if args.io_lr <= 0:
        parser.error("--io_lr must be positive.")
    needs_masks = args.input_masking or args.loss_function_masking
    if needs_masks and (args.fixed_mask is None or args.moving_mask is None):
        parser.error("Input masking/loss masking requires --fixed_mask and --moving_mask.")
    needs_segmentations = args.dice_loss_weight > 0.0
    if needs_segmentations and (args.fixed_segmentation is None or args.moving_segmentation is None):
        parser.error("Dice loss requires --fixed_segmentation and --moving_segmentation.")

    net = get_gradiconlung(
        make_sim(args.io_sim, sigma=args.sigma),
        args.intensity_conservation_loss,
        args.dice_loss_weight,
        args.loss_function_masking,
        weights_location=args.network_weights,
    )

    read = lambda path: itk.imread(path) if path is not None else None
    fixed, moving = itk.imread(args.fixed), itk.imread(args.moving)
    fixed_mask, moving_mask = read(args.fixed_mask), read(args.moving_mask)

    io_iterations = None if args.io_iterations == "None" else int(args.io_iterations)
    moving_input = preprocess(moving, moving_mask if args.input_masking else None)
    fixed_input = preprocess(fixed, fixed_mask if args.input_masking else None)

    labels = {}
    if args.loss_function_masking:
        labels.update(mask_A=moving_mask, mask_B=fixed_mask)
    if needs_segmentations:
        labels.update(segmentation_A=read(args.moving_segmentation), segmentation_B=read(args.fixed_segmentation))

    if args.svf_out:
        phi_AB, phi_BA, velocity = register_pair_svf(
            net, moving_input, fixed_input, io_iterations, args.io_lr,
            svf_steps=args.svf_steps, svf_lr=args.svf_lr, **labels)
        itk.imwrite(velocity, args.svf_out)
    else:
        phi_AB, phi_BA = register_pair_with_mask(
            net, moving_input, fixed_input, finetune_steps=io_iterations, learning_rate=args.io_lr, **labels)

    if args.transform_out:
        itk.transformwrite([phi_AB], args.transform_out)

    if args.warped_moving_out:
        moving, maybe_cast_back = maybe_cast(moving)
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
        warped_moving_image = itk.resample_image_filter(
                moving,
                transform=phi_AB,
                interpolator=interpolator,
                use_reference_image=True,
                reference_image=fixed
                )
        warped_moving_image = maybe_cast_back(warped_moving_image)
        itk.imwrite(warped_moving_image, args.warped_moving_out)

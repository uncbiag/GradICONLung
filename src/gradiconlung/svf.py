"""Stationary velocity fields (SVFs) for the network's maps.

Inside this module, velocity fields are [1, 3, D, H, W] tensors on the network grid, in mm,
with components in (z, y, x) order. ``velocity_image`` turns one into an ITK vector image.
"""
import itk
import numpy as np
import torch
from icon_registration.itk_wrapper import create_itk_transform
from icon_registration.mermaidlite import compute_warped_image_multiNC, identity_map_multiN


def identity(field, spacing):
    """Physical coordinates of every voxel of ``field``'s grid."""
    return torch.from_numpy(identity_map_multiN(field.shape, spacing)).to(field.device)


def warp(field, disp, spacing, ident=None):
    """Sample ``field`` at Id + disp (linear interpolation, border extension)."""
    if ident is None:
        ident = identity(disp, spacing)
    return compute_warped_image_multiNC(field, ident + disp, spacing, 1)


def exp(v, spacing, steps=7, ident=None):
    """Displacement of exp(v), computed by scaling and squaring."""
    if ident is None:
        ident = identity(v, spacing)
    disp = v / 2**steps
    for _ in range(steps):
        disp = disp + warp(disp, disp, spacing, ident)
    return disp


def fit_svf(disp_ab, disp_ba, spacing, steps=50, lr=0.5, tol=1e-4, exp_steps=7):
    """Fit one v such that exp(v) ~ disp_ab and exp(-v) ~ disp_ba."""
    ident = identity(disp_ab, spacing)
    v = torch.nn.Parameter(0.5 * (disp_ab - disp_ba))
    optimizer = torch.optim.Adam([v], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            forward = exp(v, spacing, exp_steps, ident)
            backward = exp(-v, spacing, exp_steps, ident)
            loss = torch.mean((forward - disp_ab) ** 2) + torch.mean((backward - disp_ba) ** 2)
            if loss.item() < tol:
                break
            if not torch.isfinite(loss):
                raise FloatingPointError("SVF fit diverged")
            loss.backward()
            optimizer.step()
    return v.detach()


def network_spacing(image, shape):
    """Physical spacing (z, y, x) of the network grid laid over ``image``.

    Matches ``icon_registration.itk_wrapper``: the network grid has the same center and extent as the image.
    """
    size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=float)[::-1]
    spacing = np.asarray(image.GetSpacing(), dtype=float)[::-1]
    return tuple(float(x) for x in spacing * size / np.asarray(shape))


def displacement_mm(phi, identity_map, spacing):
    """Convert a map in the network's [0, 1] coordinates to a displacement in mm."""
    voxels = identity_map.new_tensor(identity_map.shape[2:]).view(1, 3, 1, 1, 1) - 1
    return (phi - identity_map) * voxels * identity_map.new_tensor(spacing).view(1, 3, 1, 1, 1)


def itk_transform(v, spacing, image_A, image_B, exp_steps=7):
    """ITK transform of exp(v) that maps points of ``image_B`` to ``image_A``, like the network's phi_AB."""
    disp_mm = exp(v, spacing, exp_steps)
    disp_voxels = disp_mm / v.new_tensor(spacing).view(1, 3, 1, 1, 1)
    voxels = v.new_tensor(v.shape[2:]).view(1, 3, 1, 1, 1) - 1
    return create_itk_transform(disp_voxels / voxels, torch.zeros_like(disp_voxels), image_A, image_B)


def velocity_image(v, image):
    """ITK vector image of ``v`` on the network grid laid over ``image``.

    The vectors are in mm and in ITK physical coordinates, like an ITK displacement field.
    """
    shape = v.shape[2:]
    spacing_xyz = np.asarray(network_spacing(image, shape))[::-1]
    direction = itk.array_from_matrix(image.GetDirection())
    size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=float)
    center = np.asarray(image.GetOrigin()) + direction @ (np.asarray(image.GetSpacing()) * (size - 1) / 2)
    origin = center - direction @ (spacing_xyz * (np.asarray(shape[::-1]) - 1) / 2)

    v_index_xyz = v[0].flip(0).permute(1, 2, 3, 0).double().cpu().numpy()
    v_physical = (v_index_xyz @ direction.T).astype(np.float32)
    out = itk.image_from_array(np.ascontiguousarray(v_physical), is_vector=True)
    out.SetSpacing(spacing_xyz.tolist())
    out.SetOrigin(origin.tolist())
    out.SetDirection(image.GetDirection())
    return out


def same_physical_box(image_A, image_B, tol=1e-3):
    """Whether two images cover the same physical box, so that the network frames of both coincide."""
    def box(image):
        direction = itk.array_from_matrix(image.GetDirection())
        size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=float)
        spacing = np.asarray(image.GetSpacing())
        center = np.asarray(image.GetOrigin()) + direction @ (spacing * (size - 1) / 2)
        return np.concatenate([center, (direction * spacing * size).ravel()])

    return np.allclose(box(image_A), box(image_B), atol=tol)

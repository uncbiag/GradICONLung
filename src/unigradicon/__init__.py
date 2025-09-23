import itk
import os
from datetime import datetime

import footsteps
import numpy as np
import torch
import torch.nn.functional as F

import icon_registration as icon
import icon_registration.network_wrappers as network_wrappers
import icon_registration.networks as networks
from icon_registration import config
from icon_registration.losses import ICONLoss, to_floats
from icon_registration.mermaidlite import compute_warped_image_multiNC
import icon_registration.itk_wrapper
import gdown
import copy

input_shape = [1, 1, 175, 175, 175]


def finetune_execute(model, image_A, image_B, steps, lr=0.00002):
    state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B)
        print(loss_tuple)
        loss_tuple[0].backward()
        optimizer.step()
    with torch.no_grad():
        loss = model(image_A, image_B)
    model.load_state_dict(state_dict)
    return loss


def finetune_execute_mask(model, image_A, image_B, mask_A, mask_B, steps, lr=0.00002):
    state_dict = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss_tuple = model(image_A, image_B, mask_A=mask_A, mask_B=mask_B)
        print(loss_tuple)
        loss_tuple[0].backward()
        optimizer.step()
    with torch.no_grad():
        loss =  model(image_A, image_B, mask_A=mask_A, mask_B=mask_B)
    model.load_state_dict(state_dict)
    return loss


def register_pair(
    model, image_A, image_B, finetune_steps=None, return_artifacts=False, lr=0.00002
) :

    assert isinstance(image_A, itk.Image)
    assert isinstance(image_B, itk.Image)

    # send model to cpu or gpu depending on config- auto detects capability
    model.to(config.device)

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)
    
    assert(np.max(A_npy) != np.min(A_npy))
    assert(np.max(B_npy) != np.min(B_npy))
    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(config.device)[None, None]
    B_trch = torch.Tensor(B_npy).to(config.device)[None, None]

    shape = model.identity_map.shape

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    A_resized = F.interpolate(
        A_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    B_resized = F.interpolate(
        B_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(A_resized, B_resized)
    else:
        loss = finetune_execute(model, A_resized, B_resized, finetune_steps, lr)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_resized, B_resized)
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        icon_registration.itk_wrapper.create_itk_transform(phi_AB, model.identity_map, image_A, image_B),
        icon_registration.itk_wrapper.create_itk_transform(phi_BA, model.identity_map, image_B, image_A),
    )
    if not return_artifacts:
        return itk_transforms
    else:
        return itk_transforms + (to_floats(loss),)
    
def register_pair_with_mask(model, image_A, image_B, mask_A, mask_B, finetune_steps=None, return_artifacts=False, lr=0.00002
) :

    assert isinstance(image_A, itk.Image)
    assert isinstance(image_B, itk.Image)
    
    assert isinstance(mask_A, itk.Image)
    assert isinstance(mask_B, itk.Image)

    A_npy = np.array(image_A)
    B_npy = np.array(image_B)
    
    A_mask_npy = np.array(mask_A)
    B_mask_npy = np.array(mask_B)

    assert(np.max(A_npy) != np.min(A_npy))
    assert(np.max(B_npy) != np.min(B_npy))

    # turn images into torch Tensors: add feature and batch dimensions (each of length 1)
    A_trch = torch.Tensor(A_npy).to(config.device)[None, None]
    B_trch = torch.Tensor(B_npy).to(config.device)[None, None]
    A_mask_trch = torch.Tensor(A_mask_npy).to(config.device)[None, None]
    B_mask_trch = torch.Tensor(B_mask_npy).to(config.device)[None, None]

    shape = model.identity_map.shape

    # Here we resize the input images to the shape expected by the neural network. This affects the
    # pixel stride as well as the magnitude of the displacement vectors of the resulting
    # displacement field, which create_itk_transform will have to compensate for.
    A_resized = F.interpolate(
        A_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    B_resized = F.interpolate(
        B_trch, size=shape[2:], mode="trilinear", align_corners=False
    )
    
    A_mask_resized = F.interpolate(
        A_mask_trch, size=shape[2:], mode="nearest"
    )
    B_mask_resized = F.interpolate(
        B_mask_trch, size=shape[2:], mode="nearest"
    )
    
    if finetune_steps == 0:
        raise Exception("To indicate no finetune_steps, pass finetune_steps=None")

    if finetune_steps == None:
        with torch.no_grad():
            loss = model(A_resized, B_resized, mask_A=A_mask_resized, mask_B=B_mask_resized)  
    else:
        loss = finetune_execute_mask(model, A_resized, B_resized, A_mask_resized, B_mask_resized, finetune_steps, lr)

    # phi_AB and phi_BA are [1, 3, H, W, D] pytorch tensors representing the forward and backward
    # maps computed by the model
    if hasattr(model, "prepare_for_viz"):
        with torch.no_grad():
            model.prepare_for_viz(A_resized, B_resized)
    phi_AB = model.phi_AB(model.identity_map)
    phi_BA = model.phi_BA(model.identity_map)

    # the parameters ident, image_A, and image_B are used for their metadata
    itk_transforms = (
        icon_registration.itk_wrapper.create_itk_transform(phi_AB, model.identity_map, image_A, image_B),
        icon_registration.itk_wrapper.create_itk_transform(phi_BA, model.identity_map, image_B, image_A),
    )
    if not return_artifacts:
        return itk_transforms
    else:
        return itk_transforms + (to_floats(loss),)
    
class SSD(icon.losses.SimilarityBase):
    def __init__(self):
        super().__init__(isInterpolated=False)

    def __call__(self, warped_image_A, image_B, warped_mask_A = None, mask_B=None):
        I = warped_image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J should be the same."
        
        if mask_B is not None:
            assert mask_B.shape == J.shape, "The shape of mask and J should be the same."
        
        ssd = (I - J) ** 2
        
        mask = torch.ones_like(ssd)
        
        if warped_mask_A is not None:
            assert warped_mask_A.shape == I.shape, "The shape of mask and I should be the same."
            assert mask_B.shape == warped_mask_A.shape, "The shape of mask and warped_mask_A should be the same."
            
            warped_mask_A_clone = warped_mask_A.clone()
            warped_mask_A_clone[warped_mask_A_clone != 0] = 1
            mask = torch.logical_and(mask, warped_mask_A_clone)
        
        if mask_B is not None:
            mask_B_clone = mask_B.clone()
            mask_B_clone[mask_B_clone != 0] = 1
            mask = torch.logical_and(mask, mask_B_clone)
            
        ssd = ssd * mask

        return torch.mean(ssd)

class LNCC(icon.losses.SimilarityBase):
    def __init__(self, sigma):
        super().__init__(isInterpolated=False)
        self.sigma = sigma

    def blur(self, tensor):
        return icon.losses.gaussian_blur(tensor, self.sigma * 4 + 1, self.sigma)
    
    def calculate_lncc(self, I, J):
        return (self.blur(I * J) - (self.blur(I) * self.blur(J))) / torch.sqrt(
                (torch.relu(self.blur(I * I) - self.blur(I) ** 2) + 0.00001)
                * (torch.relu(self.blur(J * J) - self.blur(J) ** 2) + 0.00001)
            )

    def __call__(self, warped_image_A, image_B, warped_mask_A = None, mask_B=None):
        I = warped_image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J should be the same."
        
        if mask_B is not None:
            assert mask_B.shape == J.shape, "The shape of mask and J should be the same."
        
        lncc = self.calculate_lncc(I, J)
        
        mask = torch.ones_like(lncc)
        
        if warped_mask_A is not None:
            assert warped_mask_A.shape == I.shape, "The shape of mask and I should be the same."
            assert mask_B.shape == warped_mask_A.shape, "The shape of mask and warped_mask_A should be the same."
            
            warped_mask_A_clone = warped_mask_A.clone()
            warped_mask_A_clone[warped_mask_A_clone != 0] = 1
            mask = torch.logical_and(mask, warped_mask_A_clone)
        
        if mask_B is not None:
            mask_B_clone = mask_B.clone()
            mask_B_clone[mask_B_clone != 0] = 1
            mask = torch.logical_and(mask, mask_B_clone)
            
        lncc = lncc * mask

        return 1 - torch.mean(lncc)
    
class LNCCMaskGradient(icon.losses.LNCC):
    def __call__(self, warped_image_A, image_B, warped_mask_A = None, mask_B=None):
        I = warped_image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J should be the same."
        
        if mask_B is not None:
            assert mask_B.shape == J.shape, "The shape of mask and J should be the same."
        
        lncc = self.calculate_lncc(I, J)
        
        if warped_mask_A is not None and mask_B is not None:
            mask = warped_mask_A * mask_B
            
        elif mask_B is not None:
            mask = mask_B

        lncc = lncc * mask
        
        return 1 - torch.mean(lncc)

class SquaredLNCC(icon.losses.LNCC):
    def __call__(self, warped_image_A, image_B, warped_mask_A = None, mask_B=None):
        I = warped_image_A
        J = image_B
        assert I.shape == J.shape, "The shape of image I and J should be the same."
        
        if mask_B is not None:
            assert mask_B.shape == J.shape, "The shape of mask and J should be the same."
        
        lncc = self.calculate_lncc(I, J)
        
        mask = torch.ones_like(lncc)
        
        if warped_mask_A is not None:
            assert warped_mask_A.shape == I.shape, "The shape of mask and I should be the same."
            assert mask_B.shape == warped_mask_A.shape, "The shape of mask and warped_mask_A should be the same."
            
            warped_mask_A_clone = warped_mask_A.clone()
            warped_mask_A_clone[warped_mask_A_clone != 0] = 1
            mask = torch.logical_and(mask, warped_mask_A_clone)
        
        if mask_B is not None:
            mask_B_clone = mask_B.clone()
            mask_B_clone[mask_B_clone != 0] = 1
            mask = torch.logical_and(mask, mask_B_clone)
            
        lncc = lncc * mask

        return 1 - torch.mean(lncc**2)


class MINDSSC(icon.losses.SimilarityBase):
    def __init__(self, radius, dilation):
        super().__init__(isInterpolated=False)
        self.radius = radius
        self.dilation = dilation

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, torch.inf)
        return dist

    def compute_mindssc(self, img):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = self.radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(img.device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(img.device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = torch.nn.ReplicationPad3d(self.dilation)
        rpad2 = torch.nn.ReplicationPad3d(self.radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=self.dilation) - F.conv3d(rpad1(img), mshift2, dilation=self.dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind = mind / mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
        return mind

    def __call__(self, warped_image_A, image_B, warped_mask_A=None, mask_B=None):
        assert warped_image_A.shape == image_B.shape, "The shape of image_A and image_B should be the same."
        assert len(warped_image_A.shape) - 2 == 3, "The input image should be 3D."
        
        difference = self.compute_mindssc(warped_image_A) - self.compute_mindssc(image_B)
        
        mask = torch.ones_like(difference)
        
        if warped_mask_A is not None:
            assert warped_mask_A.shape == image_B.shape, "The shape of mask and I should be the same."
            assert mask_B.shape == warped_mask_A.shape, "The shape of mask and warped_mask_A should be the same."
            
            warped_mask_A_clone = warped_mask_A.clone()
            warped_mask_A_clone[warped_mask_A_clone != 0] = 1
            warped_mask_A_clone = warped_mask_A_clone.expand(-1, difference.shape[1], -1, -1, -1)

            mask = torch.logical_and(mask, warped_mask_A_clone)
        
        if mask_B is not None:
            mask_B_clone = mask_B.clone()
            mask_B_clone[mask_B_clone != 0] = 1
            mask_B_clone = mask_B_clone.expand(-1, difference.shape[1], -1, -1, -1)

            mask = torch.logical_and(mask, mask_B_clone)

        difference = difference * mask
            
        return torch.mean(difference ** 2)
    

class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda, use_label=False, apply_intensity_conservation_loss=False, use_intersection=False):
        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity
        self.use_label = use_label
        self.apply_intensity_conservation_loss = apply_intensity_conservation_loss
        self.use_intersection = use_intersection

    def forward(self, image_A, image_B, label_A=None, label_B=None, mask_A=None, mask_B=None):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        if self.use_label:
            label_A = image_A if label_A is None else label_A
            label_B = image_B if label_B is None else label_B
            assert self.identity_map.shape[2:] == label_A.shape[2:]
            assert self.identity_map.shape[2:] == label_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        if self.apply_intensity_conservation_loss:
            if len(self.input_shape) - 2 == 3:
                jacobian_slice = np.index_exp[:, :, :-1, :-1, :-1]
            elif len(self.input_shape) - 2 == 2:
                jacobian_slice = np.index_exp[:, :, :-1, :-1]
            else:
                jacobian_slice = np.index_exp[:, :, :-1]
                
            jacobian_AB = self.compute_jacobian_determinant(self.phi_AB_vectorfield)
            jacobian_BA = self.compute_jacobian_determinant(self.phi_BA_vectorfield)
            
        self.warped_image_A = compute_warped_image_multiNC(
            torch.cat([image_A, inbounds_tag], axis=1) if inbounds_tag is not None else image_A,
            self.phi_AB_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        self.warped_image_B = compute_warped_image_multiNC(
            torch.cat([image_B, inbounds_tag], axis=1) if inbounds_tag is not None else image_B,
            self.phi_BA_vectorfield,
            self.spacing,
            1,
            zero_boundary=True
        )
        
        if mask_A is not None and self.use_intersection:
            mask_A[mask_A != 0] = 1
            self.warped_mask_A = compute_warped_image_multiNC(
                torch.cat([mask_A, inbounds_tag], axis=1) if inbounds_tag is not None else mask_A,
                self.phi_AB_vectorfield,
                self.spacing,
                1,
            )
        else:
            self.warped_mask_A = None
        
        if mask_B is not None and self.use_intersection:
            mask_B[mask_B != 0] = 1
            self.warped_mask_B = compute_warped_image_multiNC(
                torch.cat([mask_B, inbounds_tag], axis=1) if inbounds_tag is not None else mask_B,
                self.phi_BA_vectorfield,
                self.spacing,
                1,
            )
        else:
            self.warped_mask_B = None
            
        if self.use_label:
            self.warped_label_A = compute_warped_image_multiNC(
                torch.cat([label_A, inbounds_tag], axis=1) if inbounds_tag is not None else label_A,
                self.phi_AB_vectorfield,
                self.spacing,
                1,
            )
            
            self.warped_label_B = compute_warped_image_multiNC(
                torch.cat([label_B, inbounds_tag], axis=1) if inbounds_tag is not None else label_B,
                self.phi_BA_vectorfield,
                self.spacing,
                1,
            )
            
            self.warped_loss_input_A = self.warped_label_A
            self.warped_loss_input_B = self.warped_label_B
        else:
            self.warped_loss_input_A = self.warped_image_A
            self.warped_loss_input_B = self.warped_image_B
            
              
        if self.apply_intensity_conservation_loss:
            self.warped_loss_input_A_jacob = self.warped_loss_input_A[jacobian_slice] * jacobian_AB
            self.warped_loss_input_B_jacob = self.warped_loss_input_B[jacobian_slice] * jacobian_BA
            similarity_loss = self.similarity(
                self.warped_loss_input_B_jacob, 
                image_A[jacobian_slice], 
                self.warped_mask_B[jacobian_slice] if self.warped_mask_B is not None else None,
                mask_A[jacobian_slice] if mask_A is not None else None
            ) + self.similarity(
                self.warped_loss_input_A_jacob, 
                image_B[jacobian_slice], 
                self.warped_mask_A[jacobian_slice] if self.warped_mask_A is not None else None,
                mask_B[jacobian_slice] if mask_B is not None else None
            )
        else:
            similarity_loss = self.similarity(self.warped_loss_input_A, image_B, self.warped_mask_A, mask_B) + self.similarity(self.warped_loss_input_B, image_A, self.warped_mask_B, mask_A)

        if len(self.input_shape) - 2 == 3:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2, ::2]
        elif len(self.input_shape) - 2 == 2:
            Iepsilon = (
                self.identity_map
                + 2 * torch.randn(*self.identity_map.shape).to(config.device)
                / self.identity_map.shape[-1]
            )[:, :, ::2, ::2]

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return icon.losses.ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            icon.losses.flips(self.phi_BA_vectorfield),
        )

    def compute_jacobian_determinant(self, phi):
        if len(phi.size()) == 4:
            du = phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]
            dv = phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]
            dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
            dA = dA[:, None, :]
            return (
                dA * (self.identity_map.shape[2] - 1) * (self.identity_map.shape[3] - 1)
            )
        if len(phi.size()) == 5:
            a = phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]
            b = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]
            c = phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]

            dV = torch.sum(torch.cross(a, b, 1) * c, dim=1, keepdim=True)
            dV = (
                dV
                * (self.identity_map.shape[2] - 1)
                * (self.identity_map.shape[3] - 1)
                * (self.identity_map.shape[4] - 1)
            )
            return dV
    
    def clean(self):
        del self.phi_AB, self.phi_BA, self.phi_AB_vectorfield, self.phi_BA_vectorfield, self.warped_image_A, self.warped_image_B
        if self.use_label:
            del self.warped_label_A, self.warped_label_B

def make_network(input_shape, include_last_step=False, lmbda=1.5, loss_fn=icon.LNCC(sigma=5), use_label=False, apply_intensity_conservation_loss=False, use_intersection=False):
    dimension = len(input_shape) - 2
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))

    for _ in range(2):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimension),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension))
        )
    if include_last_step:
        inner_net = icon.TwoStepRegistration(inner_net, icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimension)))

    net = GradientICONSparse(inner_net, loss_fn, lmbda=lmbda, use_label=use_label, apply_intensity_conservation_loss=apply_intensity_conservation_loss, use_intersection=use_intersection)
    net.assign_identity_map(input_shape)
    return net

def make_sim(similarity, sigma=5):
    if similarity == "lncc":
        return LNCC(sigma=sigma)
    elif similarity == "lncc_mask_gradient":
        return LNCCMaskGradient(sigma=sigma)
    elif similarity == "ssd":
        return SSD()
    elif similarity == "lncc2":
        return SquaredLNCC(sigma=sigma)
    elif similarity == "mind":
        return MINDSSC(radius=2, dilation=2)
    else:
        raise ValueError(f"Similarity measure {similarity} not recognized. Choose from [lncc, lncc2, mind].")

def get_multigradicon(loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, use_intersection=False):
    net = make_network(input_shape, include_last_step=True, loss_fn=loss_fn, apply_intensity_conservation_loss=apply_intensity_conservation_loss, use_intersection=use_intersection)
    from os.path import exists
    weights_location = "network_weights/multigradicon1.0/Step_2_final.trch"
    if not exists(weights_location):
        print("Downloading pretrained multigradicon model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/uniGradICON/releases/download/multigradicon_weights/Step_2_final.trch"
        os.makedirs("network_weights/multigradicon1.0/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    print(f"Loading weights from {weights_location}")
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=True)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net

def get_unigradicon(loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, use_intersection=False):
    net = make_network(input_shape, include_last_step=True, loss_fn=loss_fn, apply_intensity_conservation_loss=apply_intensity_conservation_loss, use_intersection=use_intersection)
    from os.path import exists
    weights_location = "network_weights/unigradicon1.0/Step_2_final.trch"
    if not exists(weights_location):
        print("Downloading pretrained unigradicon model")
        import urllib.request
        import os
        download_path = "https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch"
        os.makedirs("network_weights/unigradicon1.0/", exist_ok=True)
        urllib.request.urlretrieve(download_path, weights_location)
    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=True)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net


def get_gradicon_lung(loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, use_intersection=False):
    net = make_network(input_shape, include_last_step=True, loss_fn=loss_fn, apply_intensity_conservation_loss=apply_intensity_conservation_loss, use_intersection=use_intersection)
    from os.path import exists
    weights_location = "network_weights/gradicon_lung1.0/Step_2_final.trch"
    if not exists(weights_location):
        print("Downloading pretrained gradicon lung model")
        import os
        os.makedirs("network_weights/gradicon_lung1.0/", exist_ok=True)
        gdown.download(id="1UZjF0UIiMBYM9cqrpCihiPEJWD5M-Njk", output=weights_location, quiet=False)

    trained_weights = torch.load(weights_location, map_location=torch.device("cpu"), weights_only=False)
    net.regis_net.load_state_dict(trained_weights)
    net.to(config.device)
    net.eval()
    return net


def get_model_from_model_zoo(model_name="unigradicon", loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, use_intersection=False):
    if model_name == "unigradicon":
        return get_unigradicon(loss_fn, apply_intensity_conservation_loss, use_intersection)
    elif model_name == "multigradicon":
        return get_multigradicon(loss_fn, apply_intensity_conservation_loss, use_intersection)
    elif model_name == "gradiconlung":
        return get_gradicon_lung(loss_fn, apply_intensity_conservation_loss, use_intersection)
    else:
        raise ValueError(f"Model {model_name} not recognized. Choose from [unigradicon, multigradicon].")

def quantile(arr: torch.Tensor, q):
    arr = arr.flatten()
    l = len(arr)
    return torch.kthvalue(arr, int(q * l)).values

def apply_mask(image, segmentation):
    segmentation_cast_filter = itk.CastImageFilter[type(segmentation),
                                            itk.Image.F3].New()
    segmentation_cast_filter.SetInput(segmentation)
    segmentation_cast_filter.Update()
    segmentation = segmentation_cast_filter.GetOutput()
    mask_filter = itk.MultiplyImageFilter[itk.Image.F3, itk.Image.F3,
                                    itk.Image.F3].New()

    mask_filter.SetInput1(image)
    mask_filter.SetInput2(segmentation)
    mask_filter.Update()

    return mask_filter.GetOutput()

def preprocess(image, modality="ct", segmentation=None):
    if modality == "ct":
        min_ = -1000
        max_ = 1000
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    elif modality == "mri":
        image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
        min_, _ = itk.image_intensity_min_max(image)
        max_ = quantile(torch.tensor(np.array(image)), .99).item()
        image = itk.clamp_image_filter(image, Bounds=(min_, max_))
    else:
        raise ValueError(f"{modality} not recognized. Use 'ct' or 'mri'.")

    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_)) 

    if segmentation is not None:
        image = apply_mask(image, segmentation)
    return image

def main():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Register two images using unigradicon.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--fixed_modality", required=True,
                         type=str, help="The modality of the fixed image. Should be 'ct' or 'mri'.")
    parser.add_argument("--moving_modality", required=True,
                         type=str, help="The modality of the moving image. Should be 'ct' or 'mri'.")
    parser.add_argument("--fixed_segmentation", required=False,
                         type=str, help="The path of the segmentation map of the fixed image.")
    parser.add_argument("--moving_segmentation", required=False,
                         type=str, help="The path of the segmentation map of the moving image.")
    parser.add_argument("--transform_out", required=True,
                         type=str, help="The path to save the transform.")
    parser.add_argument("--warped_moving_out", required=False,
                        default=None, type=str, help="The path to save the warped image.")
    parser.add_argument("--io_iterations", required=False,
                         default="50", help="The number of IO iterations. Default is 50. Set to 'None' to disable IO.")
    parser.add_argument("--lr", required=False,
                         default="0.00002", type=float, help="The learning rate for IO. Default is 0.00002.")
    parser.add_argument("--sigma", required=False,
                         default="5", type=int, help="Sigma for LNCC and LNCC2. Default is 5.")
    parser.add_argument("--io_sim", required=False,
                         default="lncc", help="The similarity measure used in IO. Default is LNCC. Choose from [lncc, lncc2, mind].")
    parser.add_argument("--model", required=False,
                         default="unigradicon", help="The model to load. Default is unigradicon. Choose from [unigradicon, multigradicon, gradiconlung].")
    parser.add_argument("--loss_function_masking", required=False,
                         action="store_true", help="Apply loss function masking using the provided segmentations. \
                             If not set, segmentations will instead be used to mask out the images before registration.")
    parser.add_argument("--loss_function_masking_intersection", required=False,
                         action="store_true", help="Apply loss function masking using the intersection of the warped and fixed segmentations. \
                             If not set, segmentations will instead be used to mask out the images before registration.")
    parser.add_argument("--intensity_conservation_loss", required=False,
                            action="store_true", help="Enable determinant-based intensity correction in the loss \
                            function for mass-conserving registration. Applicable only for CT modality where -1000 HU represents air.")

    args = parser.parse_args()
    
    if args.intensity_conservation_loss:
        if args.fixed_modality != "ct" or args.moving_modality != "ct":
            raise ValueError("Intensity conservation loss is only supported for CT images.")

    net = get_model_from_model_zoo(args.model, make_sim(args.io_sim, args.sigma), args.intensity_conservation_loss, args.loss_function_masking_intersection)

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)
    
    if args.fixed_segmentation is not None:
        fixed_segmentation = itk.imread(args.fixed_segmentation)
    else:
        fixed_segmentation = None
    
    if args.moving_segmentation is not None:
        moving_segmentation = itk.imread(args.moving_segmentation)
    else:
        moving_segmentation = None
        
    if args.io_iterations == "None":
        io_iterations = None
    else:
        io_iterations = int(args.io_iterations)

    if args.loss_function_masking or args.loss_function_masking_intersection:
        phi_AB, phi_BA = register_pair_with_mask(
            net,
            preprocess(moving, args.moving_modality), 
            preprocess(fixed, args.fixed_modality),
            moving_segmentation,
            fixed_segmentation,
            finetune_steps=io_iterations,
            lr=args.lr)

    else:
        phi_AB, phi_BA = register_pair(
            net,
            preprocess(moving, args.moving_modality, moving_segmentation), 
            preprocess(fixed, args.fixed_modality, fixed_segmentation), 
            finetune_steps=io_iterations,
            lr=args.lr)

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

def warp_command():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Warp an image with given transformation.")
    parser.add_argument("--fixed", required=True, type=str,
                         help="The path of the fixed image.")
    parser.add_argument("--moving", required=True, type=str,
                         help="The path of the moving image.")
    parser.add_argument("--transform")
    parser.add_argument("--warped_moving_out", required=True)
    parser.add_argument('--nearest_neighbor', action='store_true')
    parser.add_argument('--linear', action='store_true')

    args = parser.parse_args()

    fixed = itk.imread(args.fixed)
    moving = itk.imread(args.moving)
    if not args.transform:
        phi_AB = itk.IdentityTransform[itk.D, 3].New()
    else:
        phi_AB = itk.transformread(args.transform)[0]

    moving, maybe_cast_back = maybe_cast(moving)

    if args.linear:
        interpolator = itk.LinearInterpolateImageFunction.New(moving)
    elif args.nearest_neighbor:
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(moving)
    else:
        raise Exception("Specify --nearest_neighbor or --linear")
    warped_moving_image = itk.resample_image_filter(
            moving,
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=fixed
            )
    
    warped_moving_image = maybe_cast_back(warped_moving_image)

    itk.imwrite(warped_moving_image, args.warped_moving_out)

def maybe_cast(img: itk.Image):
    """
    If an itk image is of a type that can't be used with InterpolateImageFunctions, cast it 
    and be able to cast it back
    """
    maybe_cast_back = lambda x: x

    if str((type(img), itk.D)) not in itk.NearestNeighborInterpolateImageFunction.GetTypesAsList():

        if type(img) in (itk.Image[itk.ULL, 3], itk.Image[itk.UL, 3]):
            raise Exception("Label maps of type unsigned long may have values that cannot be represented in a double")
 
        maybe_cast_back = itk.CastImageFilter[itk.Image[itk.D, 3], type(img)].New()

        img = itk.CastImageFilter[type(img), itk.Image[itk.D, 3]].New()(img)

    return img, maybe_cast_back

def compute_jacobian_map_command():
    import itk
    import argparse
    parser = argparse.ArgumentParser(description="Compute the Jacobian map of a given transform.")
    parser.add_argument("--transform", required=True, type=str,
                            help="The path to the transform file.")
    parser.add_argument("--fixed", required=True, type=str,
                            help="The path to the fixed image that has been used in the registration.")
    parser.add_argument("--jacob", required=True, default=None, help="The path to the output Jacobian map, \
                        e.g. /path/to/output_jacobian.nii.gz")
    parser.add_argument("--log_jacob", required=False, default=None, help="The path to the output log Jacobian map. \
                            If not specified, the log Jacobian map will not be saved.")
    args = parser.parse_args()

    transform_file = args.transform
    fixed_img_file = args.fixed
    transform = itk.transformread(transform_file)[0]
    jacob = itk.displacement_field_jacobian_determinant_filter(
        itk.transform_to_displacement_field_filter(
            transform,
            reference_image=itk.imread(fixed_img_file),
            use_reference_image=True
        )
    )
    itk.imwrite(jacob, args.jacob)

    if args.log_jacob is not None:
        log_jacob = itk.LogImageFilter.New(jacob)
        log_jacob.Update()
        log_jacob = log_jacob.GetOutput()
        itk.imwrite(log_jacob, args.log_jacob)

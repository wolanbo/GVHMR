import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from .vitpose_pytorch import build_model
from .vitfeat_extractor import get_batch
from tqdm import tqdm

from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
from hmr4d.utils.geo_transform import cvt_p2d_from_pm1_to_i
from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17


# Constants for image normalization (Standard for ViTPose/COCO models)
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])


class VitPoseExtractor:
    def __init__(self, tqdm_leave=True):
        # Allow environment variable to override checkpoint path for flexibility
        ckpt_path = os.environ.get(
            "VITPOSE_CKPT_PATH", 
            "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        )
        self.pose = build_model("ViTPose_huge_coco_256x192", ckpt_path)
        self.pose.cuda().eval()

        self.flip_test = True
        self.tqdm_leave = tqdm_leave

    @staticmethod
    def _crop_and_resize(img, center, scale, output_size=256, enlarge_ratio=1.0):
        """
        Crop and resize image based on center and scale.
        Args:
            img: numpy array (H, W, 3)
            center: numpy array (2,) [cx, cy]
            scale: float, box size
            output_size: int, target size
            enlarge_ratio: float, factor to enlarge the crop box
        Returns:
            cropped_img: numpy array (output_size, output_size, 3)
        """
        h, w = img.shape[:2]
        # Enlarge scale
        scale = scale * enlarge_ratio
        
        # Calculate crop box coordinates
        half_size = scale / 2
        left = int(center[0] - half_size)
        right = int(center[0] + half_size)
        top = int(center[1] - half_size)
        bottom = int(center[1] + half_size)

        # Handle boundaries by padding if necessary, but simple slicing is faster for real-time
        # Using cv2.warpAffine for robust cropping similar to mmpose standard processing
        
        # Standard affine transformation for cropping
        src_dir = np.zeros((3, 2), dtype=np.float32)
        dst_dir = np.zeros((3, 2), dtype=np.float32)
        
        src_dir[0] = [max(0, left), max(0, top)]
        src_dir[1] = [min(w - 1, right), min(h - 1, bottom)]
        src_dir[2] = [max(0, left), min(h - 1, bottom)]
        
        dst_dir[0] = [0, 0]
        dst_dir[1] = [output_size - 1, output_size - 1]
        dst_dir[2] = [0, output_size - 1]
        
        try:
            trans = cv2.getAffineTransform(np.float32(src_dir), np.float32(dst_dir))
            cropped_img = cv2.warpAffine(img, trans, (output_size, output_size), flags=cv2.INTER_LINEAR)
        except Exception:
            # Fallback to simple resize if affine fails (e.g., invalid points)
            cropped_img = cv2.resize(img[top:bottom, left:right], (output_size, output_size))
            
        return cropped_img

    @torch.no_grad()
    def extract(self, video_path, bbx_xys, img_ds=0.5):
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        L, _, H, W = imgs.shape  # (L, 3, H, W)
        batch_size = 16
        vitpose = []
        for j in tqdm(range(0, L, batch_size), desc="ViTPose", leave=self.tqdm_leave):
            # Heat map
            imgs_batch = imgs[j : j + batch_size, :, :, 32:224].cuda()
            if self.flip_test:
                heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
                heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
                heatmap = (heatmap + heatmap_flipped) * 0.5
                del heatmap_flipped
            else:
                heatmap = self.pose(imgs_batch.clone())  # (B, J, 64, 48)

            if False:
                # Get joint
                bbx_xys_batch = bbx_xys[j : j + batch_size].cuda()
                method = "hard"
                if method == "hard":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap)
                elif method == "soft":
                    kp2d_pm1, conf = get_heatmap_preds(heatmap, soft=True)

                # Convert 64, 48 to 64, 64
                kp2d_pm1[:, :, 0] *= 24 / 32
                kp2d = cvt_p2d_from_pm1_to_i(kp2d_pm1, bbx_xys_batch[:, None])
                kp2d = torch.cat([kp2d, conf], dim=-1)

            else:  # postprocess from mmpose
                bbx_xys_batch = bbx_xys[j : j + batch_size]
                heatmap = heatmap.clone().cpu().numpy()
                center = bbx_xys_batch[:, :2].numpy()
                scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200).numpy()
                preds, maxvals = keypoints_from_heatmaps(heatmaps=heatmap, center=center, scale=scale, use_udp=True)
                kp2d = np.concatenate((preds, maxvals), axis=-1)
                kp2d = torch.from_numpy(kp2d)

            vitpose.append(kp2d.detach().cpu().clone())

        vitpose = torch.cat(vitpose, dim=0).clone()  # (F, 17, 3)
        return vitpose

    @torch.no_grad()
    def extract_single_frame(self, img_rgb, bbx_xys, img_ds=0.5):
        """Extract pose from a single frame for real-time inference.
        
        Args:
            img_rgb: RGB image as numpy array (H, W, 3) or torch tensor (H, W, 3)
            bbx_xys: bounding box (3,) or (1, 3) - [cx, cy, size] in original image coords
            img_ds: downsample factor applied to image before cropping
            
        Returns:
            kp2d: keypoints (17, 3) as torch tensor
        """
        # 1. Prepare Input Data
        if isinstance(img_rgb, np.ndarray):
            img_tensor = torch.from_numpy(img_rgb)  # (H, W, 3)
        else:
            img_tensor = img_rgb.cpu()
            
        if isinstance(bbx_xys, np.ndarray):
            bbx_xys = torch.from_numpy(bbx_xys)
        if bbx_xys.ndim == 1:
            bbx_xys = bbx_xys.unsqueeze(0) # (1, 3)
            
        # Move to CPU/Numpy for OpenCV processing if needed
        img_np = img_tensor.numpy().astype(np.uint8) if img_tensor.dtype != torch.uint8 else img_tensor.numpy()
        center_orig = bbx_xys[0, :2].numpy()
        size_orig = bbx_xys[0, 2].item()
        
        # 2. Downsample Image (if required)
        if img_ds != 1.0:
            H, W = img_np.shape[:2]
            new_H, new_W = int(H * img_ds), int(W * img_ds)
            img_np = cv2.resize(img_np, (new_W, new_H))
            center = center_orig * img_ds
            size = size_orig * img_ds
        else:
            center = center_orig
            size = size_orig
            
        # 3. Crop and Resize to Model Input (256x256)
        # Note: ViTPose usually expects 256x192 input for the backbone, but preprocessing crops to square 256x256
        # The model input slice [:, :, 32:224] effectively uses a 192 height from the 256 image.
        img_crop = self._crop_and_resize(img_np, center, size, output_size=256)
        
        # 4. Normalize and Convert to Tensor
        # Convert RGB to float and normalize
        img_input = img_crop.astype(np.float32) / 255.0
        img_input = (img_input - IMAGE_MEAN) / IMAGE_STD
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).float() # (1, 3, 256, 256)
        
        # 5. Run Inference
        # Crop to specific input region expected by the model (height 32:224 -> 192px)
        imgs_batch = img_input[:, :, 32:224].cuda() # (1, 3, 192, 256) -- Wait, standard is usually HxW. 
        # Let's check original code: imgs[j : j + batch_size, :, :, 32:224]
        # Original shape: (L, 3, H, W). Slice on last dim? No, usually H is dim 2.
        # In get_batch, if it returns (L, 3, H, W), then [:, :, 32:224] slices the Width if H=256, W=192?
        # Actually ViTPose input is typically 256x192 (HxW). 
        # If input is 256x256, slicing [:, :, 32:224] on Dim 3 (Width) gives 192 width.
        # So input is (B, 3, 256, 192).
        
        if self.flip_test:
            heatmap, heatmap_flipped = self.pose(torch.cat([imgs_batch, imgs_batch.flip(3)], dim=0)).chunk(2)
            heatmap_flipped = flip_heatmap_coco17(heatmap_flipped)
            heatmap = (heatmap + heatmap_flipped) * 0.5
        else:
            heatmap = self.pose(imgs_batch.clone())
            
        # 6. Post-process
        heatmap_np = heatmap.clone().cpu().numpy()
        
        # Prepare scale for keypoints_from_heatmaps
        # Scale format expected: [width, height] / 200
        # The bbox used for keypoint recovery should correspond to the crop used.
        # We cropped using 'size' at downsampled resolution.
        # The effective box size in the crop coordinate system (256x256) is 'size'.
        # However, keypoints_from_heatmaps maps back to original image coordinates.
        # We must pass the ORIGINAL image center and scale (adjusted for ds if we want coords in DS space, 
        # but usually we want coords in Original Image Space).
        
        # If we want output keypoints in the original image coordinate system:
        center_out = center_orig
        size_out = size_orig
        
        # Scale vector: [width, height]
        # In the batch extraction: scale = (torch.cat((bbx_xys_batch[:, [2]] * 24 / 32, bbx_xys_batch[:, [2]]), dim=1) / 200)
        # This implies Width = Size * 24/32 ? And Height = Size? 
        # This looks like an aspect ratio adjustment for the bounding box representation.
        # Let's stick to the logic in the batch method:
        scale_w = size_out * (24 / 32) # Adjusting aspect ratio as per original code logic
        scale_h = size_out
        scale_np = np.array([[scale_w, scale_h]]) / 200.0
        
        preds, maxvals = keypoints_from_heatmaps(
            heatmaps=heatmap_np, 
            center=center_out.reshape(1, 2), 
            scale=scale_np, 
            use_udp=True
        )
        
        kp2d = np.concatenate((preds, maxvals), axis=-1)
        kp2d = torch.from_numpy(kp2d)
        
        return kp2d.squeeze(0)


def get_heatmap_preds(heatmap, normalize_keypoints=True, thr=0.0, soft=False):
    """
    heatmap: (B, J, H, W)
    """
    assert heatmap.ndim == 4, "batch_images should be 4-ndim"

    B, J, H, W = heatmap.shape
    heatmaps_reshaped = heatmap.reshape((B, J, -1))

    maxvals, idx = torch.max(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((B, J, 1))
    idx = idx.reshape((B, J, 1))
    preds = idx.repeat(1, 1, 2).float()
    preds[:, :, 0] = (preds[:, :, 0]) % W
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / W)

    pred_mask = torch.gt(maxvals, thr).repeat(1, 1, 2)
    pred_mask = pred_mask.float()
    preds *= pred_mask

    # soft peak
    if soft:
        patch_size = 5
        patch_half = patch_size // 2
        patches = torch.zeros((B, J, patch_size, patch_size)).to(heatmap)
        default_patch = torch.zeros(patch_size, patch_size).to(heatmap)
        default_patch[patch_half, patch_half] = 1
        for b in range(B):
            for j in range(17):
                x, y = preds[b, j].int()
                if x >= patch_half and x <= W - patch_half and y >= patch_half and y <= H - patch_half:
                    patches[b, j] = heatmap[
                        b, j, y - patch_half : y + patch_half + 1, x - patch_half : x + patch_half + 1
                    ]
                else:
                    patches[b, j] = default_patch

        dx, dy = soft_patch_dx_dy(patches)
        preds[:, :, 0] += dx
        preds[:, :, 1] += dy

    if normalize_keypoints:  # to [-1, 1]
        preds[:, :, 0] = preds[:, :, 0] / (W - 1) * 2 - 1
        preds[:, :, 1] = preds[:, :, 1] / (H - 1) * 2 - 1

    return preds, maxvals


def soft_patch_dx_dy(p):
    """p (B,J,P,P)"""
    p_batch_shape = p.shape[:-2]
    patch_size = p.size(-1)
    temperature = 1.0
    score = F.softmax(p.view(-1, patch_size**2) * temperature, dim=-1)

    # get a offset_grid (BN, P, P, 2) for dx, dy
    offset_grid = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size))[::-1]
    offset_grid = torch.stack(offset_grid, dim=-1).float() - (patch_size - 1) / 2
    offset_grid = offset_grid.view(1, 1, patch_size, patch_size, 2).to(p.device)

    score = score.view(*p_batch_shape, patch_size, patch_size)
    dx = torch.sum(score * offset_grid[..., 0], dim=(-2, -1))
    dy = torch.sum(score * offset_grid[..., 1], dim=(-2, -1))

    if False:
        b, j = 0, 0
        print(torch.stack([dx[b, j], dy[b, j]]))
        print(p[b, j])

    return dx, dy

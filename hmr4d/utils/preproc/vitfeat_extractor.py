import torch
import torch.nn.functional as F
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "video":
        imgs = read_video_np(input_path, scale=img_ds)
    elif path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path

    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    if True:
        gt_bbx_size_ds = gt_bbx_size * img_ds
        ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
        imgs = np.stack(
            [
                # gaussian(v, sigma=(d - 1) / 2, channel_axis=2, preserve_range=True) if d > 1.1 else v
                cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
                for v, d in zip(imgs, ds_factors)
            ]
        )

    # Output
    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256
    return imgs, bbx_xys


class Extractor:
    def __init__(self, tqdm_leave=True):
        self.extractor: HMR2 = load_hmr2().cuda().eval()
        self.tqdm_leave = tqdm_leave

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        imgs = imgs.cuda()
        batch_size = 16  # 5GB GPU memory, occupies all CUDA cores of 3090
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size]

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features
    
    @torch.no_grad()
    def extract_single_frame_features(self, img_rgb, bbx_xys, img_ds=0.5):
        """Extract visual features from a single frame
            
        Args:
            img_rgb: RGB image as numpy array (H, W, 3) or torch tensor
            bbx_xys: bounding box (1, 3) or (3,) - [cx, cy, size]
            img_ds: downsample factor
                
        Returns:
            features: visual features (1024,) for single frame
        """
        # Prepare input
        if isinstance(img_rgb, np.ndarray):
            img_tensor = torch.from_numpy(img_rgb).unsqueeze(0)  # (1, H, W, 3)
        else:
            img_tensor = img_rgb.unsqueeze(0) if img_rgb.ndim == 3 else img_rgb
            
        if isinstance(bbx_xys, np.ndarray):
            bbx_xys = torch.from_numpy(bbx_xys)
        if bbx_xys.ndim == 1:
            bbx_xys = bbx_xys.unsqueeze(0)
            
        # Downsample image
        if img_ds != 1.0:
            H, W = img_tensor.shape[1], img_tensor.shape[2]
            new_H, new_W = int(H * img_ds), int(W * img_ds)
            img_tensor = F.interpolate(
                img_tensor.permute(0, 3, 1, 2), 
                size=(new_H, new_W), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
            
        # Crop and prepare for model
        gt_center = bbx_xys[:, :2] * img_ds
        gt_bbx_size = bbx_xys[:, 2] * img_ds
            
        imgs_list = []
        for i in range(len(img_tensor)):
            img_np = img_tensor[i].numpy().astype(np.uint8)
            img_crop, _ = crop_and_resize(
                img_np,
                gt_center[i].numpy(),
                gt_bbx_size[i].item(),
                256,
                enlarge_ratio=1.0,
            )
            imgs_list.append(img_crop)
            
        imgs = torch.from_numpy(np.stack(imgs_list))  # (1, 256, 256, 3)
        imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (1, 3, 256, 256)
            
        # Run inference
        imgs_cuda = imgs.cuda()
        features = self.extractor({"img": imgs_cuda})
            
        # Return features (remove batch dimension if single frame)
        return features.squeeze(0) if features.shape[0] == 1 else features
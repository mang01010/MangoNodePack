# MangoImageLoader.py
import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers

class MangoImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        subfolders = []
        if output_dir and os.path.exists(output_dir):
            # Find all subfolders under the output directory
            for root, dirs, files in os.walk(output_dir):
                for d in dirs:
                    rel_path = os.path.relpath(os.path.join(root, d), output_dir)
                    subfolders.append(rel_path)
        subfolders = sorted(subfolders)
        if not subfolders:
            subfolders = [""]
        return {
            "required": {
                "subfolder": (subfolders, {"tooltip": "Select a subfolder from the ComfyUI output folder."})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_images"
    CATEGORY = "Mango Node Pack/Loaders"
    DESCRIPTION = "Loads all images from the specified subfolder in the ComfyUI output directory as a batch for img2img."

    def load_images(self, subfolder):
        output_dir = folder_paths.get_output_directory()
        if not output_dir:
            print("Error: Output directory not found! Check your ComfyUI setup.")
            return (torch.zeros((1, 1, 1, 1)), torch.zeros((1, 1, 1)))

        folder = os.path.join(output_dir, subfolder)
        if not os.path.exists(folder):
            print(f"Error: Subfolder '{subfolder}' not found in output directory.")
            return (torch.zeros((1, 1, 1, 1)), torch.zeros((1, 1, 1)))

        # Gather all common image files
        files = sorted(
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"))
        )

        output_images = []
        output_masks = []

        for fname in files:
            path = os.path.join(folder, fname)
            img = node_helpers.pillow(Image.open, path)
            for frame in ImageSequence.Iterator(img):
                frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
                if frame.mode == "I":
                    frame = frame.point(lambda i: i * (1 / 255))
                rgb = frame.convert("RGB")

                arr = np.array(rgb).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr)[None,]  # shape: [1, H, W, C]

                # alpha-channel → mask
                if "A" in frame.getbands():
                    alpha = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
                    mask = 1.0 - torch.from_numpy(alpha)
                else:
                    # fall back to an all-zero mask
                    h, w = rgb.size[1], rgb.size[0]
                    mask = torch.zeros((h, w), dtype=torch.float32)

                output_images.append(tensor)
                output_masks.append(mask.unsqueeze(0))
                break  # only first frame of each file

        if output_images:
            batch_images = torch.cat(output_images, dim=0)
            batch_masks  = torch.cat(output_masks, dim=0)
        else:
            # no images? return a dummy 1×1 tensor
            batch_images = torch.zeros((1, 1, 1, 1))
            batch_masks  = torch.zeros((1, 1, 1))

        return (batch_images, batch_masks)

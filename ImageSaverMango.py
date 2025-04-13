import os
import re
import json
from datetime import datetime

import numpy as np
from PIL import Image, PngImagePlugin
import folder_paths

class ImageSaverMango:

    CATEGORY = "Mango Node Pack/Metadata"
    OUTPUT_NODE = True
    FUNCTION = "save_images"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to save"}),
                "metadata": ("METADATA", {"tooltip": "Metadata dictionary from upstream (e.g. from KSampler (Mango))"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "Prefix for the saved file. Supports placeholders like %date:hhmmss%."}),
                "subdirectory_name": ("STRING", {"default": "", "tooltip": "Custom subdirectory (inside the output folder) to save the image."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ()
    DESCRIPTION = (
        "Saves images with embedded metadata. Filenames are numbered as prefix_00001, prefix_00002, etc. "
        "Also creates a single 'parameters' text chunk containing the typical KSampler info (prompt, seed, etc.)."
    )

    pattern_format = re.compile(r"(%[^%]+%)")

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.compress_level = 4
        self.prefix_append = ""

    def save_images(self, images, metadata, filename_prefix="ComfyUI", subdirectory_name="", prompt=None, extra_pnginfo=None):
        param_string = self.build_param_string(metadata)

        base_name = self.format_filename(filename_prefix, metadata) + self.prefix_append

        if subdirectory_name:
            subdirectory_name = self.format_filename(subdirectory_name, metadata)
            full_output_folder = os.path.join(self.output_dir, subdirectory_name)
        else:
            full_output_folder = self.output_dir
        os.makedirs(full_output_folder, exist_ok=True)

        base_format = "png"
        start_suffix = self.next_available_suffix(base_name, full_output_folder, base_format)

        results = []
        for i, image in enumerate(images):
            arr = self.to_uint8(image)
            pil_img = Image.fromarray(arr)
            # Build a PngInfo object.
            pnginfo = self.prepare_pnginfo(metadata, i, len(images), prompt=prompt, extra_pnginfo=extra_pnginfo)           #added prompt=prompt, extra_pnginfo=extra_pnginfo)
            pnginfo.add_text("parameters", param_string)
            suffix_number = start_suffix + i
            filename_with_suffix = f"{base_name}_{suffix_number:05d}"
            file = f"{filename_with_suffix}.{base_format}"
            full_path = os.path.join(full_output_folder, file)
            pil_img.save(full_path, pnginfo=pnginfo, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": full_output_folder,
                "type": "output",
            })

        return {"ui": {"images": results}}

    def build_param_string(self, meta):
        pos_prompt = meta.get("Positive prompt", "")
        neg_prompt = meta.get("Negative prompt", "")
        steps = meta.get("Steps", 20)
        sampler = meta.get("Sampler", "euler")
        cfg_scale = meta.get("CFG scale", 7.0)
        seed = meta.get("Seed", 0)
        scheduler = meta.get("Scheduler", "normal")
        denoise = meta.get("Denoise", 1.0)
        model_file = meta.get("Model", "UnknownModel")
        model_hash = meta.get("Model hash", "")
        lora_hashes = meta.get("Lora hashes", {})
        hashes = meta.get("Hashes", "{}")

        param_string = (
            f"{pos_prompt}\n"
            f"Negative prompt: {neg_prompt}\n"
            f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg_scale}, Seed: {seed}, "
            f"Scheduler: {scheduler}, Denoise: {denoise}, "
            f"Model: {model_file}, Model hash: {model_hash}"
        )
        if lora_hashes:
            lora_str = ", ".join(lora_hashes.values())
            param_string += f", Lora hashes: \"{lora_str}\""
        if hashes:
            param_string += f", Hashes: {hashes}"
        return param_string


    def to_uint8(self, image):
        if hasattr(image, "cpu"):
            arr = image.cpu().numpy() * 255.0
        else:
            arr = np.array(image) * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def prepare_pnginfo(self, meta_dict, index, total, prompt=None, extra_pnginfo=None):            #added prompt=None, extra_pnginfo=None
        pnginfo = PngImagePlugin.PngInfo()
        if total > 1:
            pnginfo.add_text("Batch index", str(index))
            pnginfo.add_text("Batch size", str(total))
        for key, value in meta_dict.items():
            pnginfo.add_text(str(key), str(value))
        if prompt is not None:                                                                      #added from here...
            pnginfo.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for k, v in extra_pnginfo.items():
                pnginfo.add_text(str(k), json.dumps(v))                                             #...to here
        return pnginfo

    def format_filename(self, filename, meta_dict):
        matches = self.pattern_format.findall(filename)
        now = datetime.now()
        date_table = {
            "yyyy": str(now.year),
            "yy": str(now.year)[-2:],
            "MM": str(now.month).zfill(2),
            "dd": str(now.day).zfill(2),
            "hh": str(now.hour).zfill(2),
            "mm": str(now.minute).zfill(2),
            "ss": str(now.second).zfill(2),
        }
        for match in matches:
            if match.startswith("%date"):
                parts = match.strip("%").split(":")
                fmt = parts[1] if len(parts) > 1 else "yyyyMMddhhmmss"
                for k, v in date_table.items():
                    fmt = fmt.replace(k, v)
                filename = filename.replace(match, fmt)
        return filename

    def next_available_suffix(self, base_name, folder, ext):
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.{ext}$")
        max_num = 0
        for fname in os.listdir(folder):
            m = pattern.match(fname)
            if m:
                try:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
                except Exception:
                    pass
        return max_num + 1

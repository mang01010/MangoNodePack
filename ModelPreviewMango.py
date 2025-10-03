import os
import math
import yaml
from PIL import Image
import folder_paths

def collect_dirs(key_name: str):
    """
    key_name: "checkpoints" or "loras"
    Returns a list of absolute dirs:
      1) ComfyUI/models/<key_name>
      2) For each section in extra_model_paths.yaml:
           section.base_path + '/' + section[key_name]
    """
    ROOT = os.getcwd()
    dirs = [os.path.join(ROOT, "models", key_name)]
    yaml_file = os.path.join(ROOT, "extra_model_paths.yaml")
    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # Each section is a dict with base_path and relative subdirs
        for section in cfg.values():
            base = section.get("base_path", ROOT)
            rel  = section.get(key_name)
            if rel:
                dirs.append(os.path.abspath(os.path.join(base, rel)))
    return dirs

class PreviewImageNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name":  ("STRING",),
                "lora_stack": ("LORA_STACK",),
            }
        }
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("preview",)
    FUNCTION      = "preview_images"
    CATEGORY      = "Mango Node Pack/Loaders"

    def preview_images(self, ckpt_name, lora_stack):
        # strip off extensions so "foo.ckpt" → "foo"
        base_name = os.path.splitext(ckpt_name)[0]

        exts = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]

        def find_image_in(dirs, name):
            for d in dirs:
                if not os.path.isdir(d):
                    continue
                for ext in exts:
                    candidate = os.path.join(d, name + ext)
                    if os.path.exists(candidate):
                        print(f"[PreviewImage] ✔ Found {name+ext} in {d}")
                        return candidate
            print(f"[PreviewImage] ✘ No preview for '{name}' in {dirs}")
            return None

        # build our search lists
        model_dirs = collect_dirs("checkpoints")
        lora_dirs  = collect_dirs("loras")

        imgs = []
        # 1) model preview
        mp = find_image_in(model_dirs, base_name)
        imgs.append(Image.open(mp).convert("RGB") if mp else Image.new("RGB",(512,512),(40,40,40)))

        # 2) up to 5 LoRAs
        for name, *_ in lora_stack[:5]:
            if not name:
                continue
            lp = find_image_in(lora_dirs, name)
            imgs.append(Image.open(lp).convert("RGB") if lp else Image.new("RGB",(512,512),(80,80,80)))

        # 3) composite 2×3 grid
        cols = 3
        rows = math.ceil(len(imgs)/cols)
        w, h = imgs[0].size
        grid = Image.new("RGB", (cols*w, rows*h), (20,20,20))
        for i, im in enumerate(imgs):
            grid.paste(im, ((i%cols)*w, (i//cols)*h))

        return (grid,)

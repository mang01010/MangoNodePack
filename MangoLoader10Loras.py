import os
import hashlib
import folder_paths
import comfy.sd

def short_hash(path):
    if not os.path.exists(path):
        return "no_file"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()[:10]

def load_ckpt(ckpt_name):
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint not found: {ckpt_name}")
    loaded = comfy.sd.load_checkpoint_guess_config(
        ckpt_path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    model, clip, vae = loaded[:3]
    return model, clip, vae, short_hash(ckpt_path)

def load_single_lora(model, clip, lora_name, weight):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.exists(lora_path):
        return model, clip
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, weight, weight)
    return model, clip

class MangoLoader10Loras:

    @classmethod
    def INPUT_TYPES(cls):
        ckpts = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (ckpts, {"default": ckpts[0] if ckpts else None, "tooltip": "Select a checkpoint"}),
                "LoraName1": (loras, {"default": "None", "tooltip": "LoRA 1 filename"}),
                "LoraWeight1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 1 weight"}),
                "LoraName2": (loras, {"default": "None", "tooltip": "LoRA 2 filename"}),
                "LoraWeight2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 2 weight"}),
                "LoraName3": (loras, {"default": "None", "tooltip": "LoRA 3 filename"}),
                "LoraWeight3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 3 weight"}),
                "LoraName4": (loras, {"default": "None", "tooltip": "LoRA 4 filename"}),
                "LoraWeight4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 4 weight"}),
                "LoraName5": (loras, {"default": "None", "tooltip": "LoRA 5 filename"}),
                "LoraWeight5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 5 weight"}),
                "LoraName6": (loras, {"default": "None", "tooltip": "LoRA 6 filename"}),
                "LoraWeight6": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 6 weight"}),
                "LoraName7": (loras, {"default": "None", "tooltip": "LoRA 7 filename"}),
                "LoraWeight7": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 7 weight"}),
                "LoraName8": (loras, {"default": "None", "tooltip": "LoRA 8 filename"}),
                "LoraWeight8": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 8 weight"}),
                "LoraName9": (loras, {"default": "None", "tooltip": "LoRA 9 filename"}),
                "LoraWeight9": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 9 weight"}),
                "LoraName10": (loras, {"default": "None", "tooltip": "LoRA 10 filename"}),
                "LoraWeight10": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 10 weight"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "lora_stack", "ckpt_name", "ckpt_hash")
    FUNCTION = "load_checkpoint_and_loras"
    CATEGORY = "Mango Node Pack/Loaders"

    def load_checkpoint_and_loras(self, ckpt_name, **kwargs):
        model, clip, vae, ckpt_hash = load_ckpt(ckpt_name)
        lora_stack = []
        for i in range(1, 11):
            name = kwargs.get(f"LoraName{i}")
            weight = kwargs.get(f"LoraWeight{i}", 1.0)
            if name and name != "None":
                model, clip = load_single_lora(model, clip, name, weight)
                lora_stack.append((name, weight, weight))
        return (model, clip, vae, lora_stack, ckpt_name, ckpt_hash)

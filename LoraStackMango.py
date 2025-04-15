import os
import folder_paths
import comfy.sd
import comfy.utils

def load_single_lora(model, clip, lora_name, weight):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.exists(lora_path):
        return model, clip
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, weight, weight)
    return model, clip

class LoraStackMango:

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("LORA_STACK", {"default": []}),
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
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK")
    RETURN_NAMES = ("model", "clip", "lora_stack")
    FUNCTION = "apply_loras"
    CATEGORY = "Mango Node Pack/Loaders"

    def apply_loras(self, model, clip, lora_stack, **kwargs):
        new_lora_stack = list(lora_stack) if lora_stack else []
        for i in range(1, 6):
            name = kwargs.get(f"LoraName{i}")
            weight = kwargs.get(f"LoraWeight{i}", 1.0)
            if name and name != "None":
                model, clip = load_single_lora(model, clip, name, weight)
                new_lora_stack.append((name, weight, weight))
        return (model, clip, new_lora_stack)

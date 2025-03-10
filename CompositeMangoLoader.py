import os
import hashlib
import folder_paths
import comfy.sd
import comfy.utils
import torch

def short_hash(path):
    if not os.path.exists(path):
        return "no_file"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()[:10]

def load_single_lora(model, clip, lora_name, weight):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.exists(lora_path):
        return model, clip
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    model, clip = comfy.sd.load_lora_for_models(model, clip, lora, weight, weight)
    return model, clip

class CompositeMangoLoader:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        taesd_combinations = {
            "taesd": ("taesd_encoder", "taesd_decoder"),
            "taesdxl": ("taesdxl_encoder", "taesdxl_decoder"),
            "taesd3": ("taesd3_encoder", "taesd3_decoder"),
            "taef1": ("taef1_encoder", "taef1_decoder")
        }
        for name, (enc_prefix, dec_prefix) in taesd_combinations.items():
            if any(v.startswith(enc_prefix) for v in approx_vaes) and any(v.startswith(dec_prefix) for v in approx_vaes):
                vaes.append(name)
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        enc_prefix, dec_prefix = {
            "taesd": ("taesd_encoder", "taesd_decoder"),
            "taesdxl": ("taesdxl_encoder", "taesdxl_decoder"),
            "taesd3": ("taesd3_encoder", "taesd3_decoder"),
            "taef1": ("taef1_encoder", "taef1_decoder")
        }[name]

        encoder = next(v for v in approx_vaes if v.startswith(enc_prefix))
        decoder = next(v for v in approx_vaes if v.startswith(dec_prefix))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", encoder))
        dec = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", decoder))

        for k in enc: sd[f"{enc_prefix}.{k}"] = enc[k]
        for k in dec: sd[f"{dec_prefix}.{k}"] = dec[k]

        sd.update({
            "taesd": (0.18215, 0.0),
            "taesdxl": (0.13025, 0.0),
            "taesd3": (1.5305, 0.0609),
            "taef1": (0.3611, 0.1159)
        }[name])
        sd["vae_scale"], sd["vae_shift"] = torch.tensor(sd["vae_scale"]), torch.tensor(sd["vae_shift"])
        return sd

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "type": (["sdxl", "sd3", "flux", "hunyuan_video"],),
                "vae_name": (cls.vae_list(),),
                "LoraName1": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
                "LoraWeight1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "LoraName2": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
                "LoraWeight2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "LoraName3": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
                "LoraWeight3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "LoraName4": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
                "LoraWeight4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "LoraName5": (["None"] + folder_paths.get_filename_list("loras"), {"default": "None"}),
                "LoraWeight5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "LORA_STACK", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "lora_stack", "unet_name", "unet_hash")
    FUNCTION = "load_all"
    CATEGORY = "Mango Node Pack/Loaders"

    def load_all(self, unet_name, weight_dtype, clip_name1, clip_name2, type, vae_name, device="default", **kwargs):
        # Load UNET
        model_options = {}
        if weight_dtype.startswith("fp8"):
            model_options["dtype"] = getattr(torch, f"float{weight_dtype[2:]}")
        if "fast" in weight_dtype:
            model_options["fp8_optimizations"] = True

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        unet_hash = short_hash(unet_path)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

        # Load Dual CLIP
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_type = getattr(comfy.sd.CLIPType, type.upper())

        model_options_clip = {}
        if device == "cpu":
            model_options_clip.update({"load_device": "cpu", "offload_device": "cpu"})

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            clip_type=clip_type,
            model_options=model_options_clip
        )

        # Load VAE
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            vae_sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)

        # Apply LORAs
        lora_stack = []
        for i in range(1, 6):
            lora_name = kwargs.get(f"LoraName{i}")
            lora_weight = kwargs.get(f"LoraWeight{i}", 1.0)
            if lora_name and lora_name != "None":
                model, clip = load_single_lora(model, clip, lora_name, lora_weight)
                lora_stack.append((lora_name, lora_weight, lora_weight))

        return (model, clip, vae, lora_stack, unet_name, unet_hash)

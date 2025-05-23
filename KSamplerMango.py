import os
import json
import torch
import hashlib
from comfy import samplers
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview
import folder_paths
from .MangoTriggerExporter import get_lora_metadata

LAST_USED_SEED = None

def short_sha256(file_path):
    if not os.path.exists(file_path):
        return "no_file"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def find_model_file_and_hash(model_name):
    candidate = folder_paths.get_full_path("checkpoints", model_name)
    if candidate and os.path.exists(candidate):
        return (os.path.basename(candidate), short_sha256(candidate))
    return (model_name, "no_file")


def parse_loras_from_stack(lora_stack):
    results = {}
    if not lora_stack:
        return results
    for item in lora_stack:
        if isinstance(item, (tuple, list)) and len(item) > 0:
            lora_name = item[0]
            if lora_name and lora_name != "None":
                base = os.path.basename(lora_name)
                lora_path = folder_paths.get_full_path("loras", lora_name)
                if lora_path and os.path.exists(lora_path):
                    lora_hash = short_sha256(lora_path)
                    results[base] = f"{base}: {lora_hash}"
                else:
                    results[base] = base
    return results


class BaseNode:
    pass


class KSamplerMango(BaseNode):
    CATEGORY = "Mango Node Pack/Sampling"
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent_image": ("LATENT", {"tooltip": "Latent to sample from"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reuse_seed": ("BOOLEAN", {"default": False}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (samplers.KSampler.SAMPLERS,),
                "scheduler": (samplers.SCHEDULER_NAMES,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning tensor"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning tensor"}),
                "positive_prompt_text": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt_text": ("STRING", {"default": "", "multiline": True}),
                "ckpt_name": ("STRING", {"default": "", "multiline": False, "tooltip": "Checkpoint name (from MangoLoader)"}),
                "lora_stack": ("LORA_STACK", {"tooltip": "Optional LoRA stack (from MangoLoader)"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT", "METADATA", "STRING")
    RETURN_NAMES = ("LATENT", "METADATA", "METADATA_TEXT")
    DESCRIPTION = (
        "KSampler (Mango) with live preview and cancel support, plus metadata compilation."
    )

    def sample(
        self,
        model,
        latent_image,
        seed,
        reuse_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        positive,
        negative,
        positive_prompt_text,
        negative_prompt_text,
        ckpt_name,
        lora_stack,
        prompt=None,
        extra_pnginfo=None,
        my_unique_id=None
    ):
        # Check for cancellation before starting
        comfy.model_management.throw_exception_if_processing_interrupted()

        global LAST_USED_SEED
        used_seed = LAST_USED_SEED if (reuse_seed and LAST_USED_SEED is not None) else seed

        # Determine device
        if hasattr(model, "device"):
            device = model.device
        else:
            try:
                device = model.get_model_object("model").device
            except Exception:
                device = torch.device("cpu")

        # Validate latent_image
        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            raise ValueError("Invalid latent_image input. Provide a valid LATENT node output.")
        latent_tensor = latent_image["samples"]
        noise_mask = latent_image.get("noise_mask", None)

        # Generate noise
        generator = torch.Generator(device=device).manual_seed(used_seed)
        noise = torch.randn(latent_tensor.shape, generator=generator, device=device, dtype=latent_tensor.dtype)

        # Prepare live preview callback
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Perform sampling with preview and cancellation support
        samples_out = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_tensor,
            denoise=denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=used_seed
        )
        LAST_USED_SEED = used_seed

        # Build output latent structure
        out_latent = latent_image.copy()
        out_latent["samples"] = samples_out

        # Compile metadata
        metadata = {}
        metadata["Seed"] = used_seed
        metadata["Steps"] = steps
        metadata["CFG scale"] = cfg
        metadata["Sampler"] = sampler_name
        metadata["Scheduler"] = scheduler
        metadata["Denoise"] = denoise

        used_ckpt = str(ckpt_name).strip() if ckpt_name and str(ckpt_name).strip() else "UnknownModel"
        model_filename, model_short_hash = find_model_file_and_hash(used_ckpt)
        metadata["Model"] = model_filename
        metadata["Model hash"] = model_short_hash

        metadata["Positive prompt"] = positive_prompt_text
        metadata["Negative prompt"] = negative_prompt_text

        if isinstance(lora_stack, tuple):
            lora_list = lora_stack[0]
        else:
            lora_list = lora_stack
        lora_info = parse_loras_from_stack(lora_list)
        if lora_info:
            metadata["Lora hashes"] = lora_info

        hash_dict = {"model": model_short_hash}
        if lora_list:
            for item in lora_list:
                if isinstance(item, (tuple, list)) and len(item) > 0:
                    ln = item[0]
                    if ln and ln != "None":
                        base = os.path.basename(ln)
                        lora_path = folder_paths.get_full_path("loras", ln)
                        if lora_path and os.path.exists(lora_path):
                            hash_dict[f"lora:{base}"] = short_sha256(lora_path)
        metadata["Hashes"] = json.dumps(hash_dict)

        # Include any extra PNG info
        if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
            for k, v in extra_pnginfo.items():
                s = str(v)
                metadata[k] = s if len(s) < 1000 else s[:1000] + " [truncated]"
        if prompt is not None:
            s = str(prompt)
            metadata["prompt"] = s if len(s) < 1000 else s[:1000] + " [truncated]"

        metadata_str = json.dumps(metadata, indent=2)
        return (out_latent, metadata, metadata_str)

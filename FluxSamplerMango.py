import os
import json
import hashlib
import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
import folder_paths

# Global variables for noise reuse
LAST_USED_NOISE = None
LAST_USED_NOISE_SEED = None

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

class FluxNoise:
    """
    A simple noise wrapper mimicking a random noise generator.
    """
    def __init__(self, seed, noise_tensor):
        self.seed = seed
        self.noise_tensor = noise_tensor

    def generate_noise(self, latent):
        return self.noise_tensor

def inline_basic_guider(model, conditioning):
    """
    Build a simple guider using CFGGuider with a single 'positive' conditioning.
    (Note: guidance is not embedded in conditioning here; it comes via flux_guidance.)
    """
    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": conditioning})
    return guider

def inline_basic_scheduler(model, scheduler, steps, denoise):
    """
    Compute sigmas using the same logic as BasicScheduler.
    """
    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return torch.FloatTensor([])
        total_steps = int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps
    ).cpu()
    sigmas = sigmas[-(steps + 1):]
    return sigmas

class FluxSamplerMango:

    CATEGORY = "Mango Node Pack/Sampling"
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "flux_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "lora_stack": ("LORA_STACK",),
                "latent_image": ("LATENT",),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "unet_name": ("STRING", {"default": "", "tooltip": "UNET/Checkpoint name"}),
                "unet_hash": ("STRING", {"default": "", "tooltip": "UNET hash from CompositeMangoLoader"}),
                "noise_seed": ("INT", {
                    "default": 1234,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True
                }),
                "reuse_noise": ("BOOLEAN", {"default": False}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "METADATA", "STRING")
    RETURN_NAMES = ("output", "METADATA", "METADATA_TEXT")

    def sample(
        self,
        model,
        conditioning,
        flux_guidance,
        lora_stack,
        latent_image,
        prompt_text,
        unet_name,
        unet_hash,
        noise_seed,
        reuse_noise,
        sampler_name,
        scheduler,
        steps,
        denoise,
    ):
        global LAST_USED_NOISE, LAST_USED_NOISE_SEED

        # Determine device from the model
        if hasattr(model, "device"):
            device = model.device
        else:
            try:
                device = model.get_model_object("model").device
            except Exception:
                device = torch.device("cpu")

        if not isinstance(latent_image, dict) or "samples" not in latent_image:
            raise ValueError("Invalid latent_image input. Provide a valid LATENT node output.")
        latent = latent_image.copy()
        original_tensor = latent["samples"]

        # Build the guider using the provided conditioning (without altering resolution)
        guider = inline_basic_guider(model, conditioning)
        # Fix latent channels using the model directly (as in SamplerCustomAdvanced)
        fixed_latent = comfy.sample.fix_empty_latent_channels(model, original_tensor)
        latent["samples"] = fixed_latent

        if reuse_noise and (LAST_USED_NOISE is not None):
            used_noise_tensor = LAST_USED_NOISE
            used_seed = LAST_USED_NOISE_SEED
        else:
            used_seed = noise_seed
            generator = torch.Generator(device=device).manual_seed(used_seed)
            used_noise_tensor = torch.randn(fixed_latent.shape, generator=generator, device=device, dtype=fixed_latent.dtype)
            LAST_USED_NOISE = used_noise_tensor
            LAST_USED_NOISE_SEED = used_seed

        noise = FluxNoise(used_seed, used_noise_tensor)
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = inline_basic_scheduler(model, scheduler, steps, denoise)
        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
        noise_mask = latent_image.get("noise_mask") if "noise_mask" in latent_image else None
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = guider.sample(
            noise.generate_noise(latent),
            fixed_latent,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        metadata = {}
        metadata["Seed"] = used_seed
        metadata["Steps"] = steps
        metadata["Sampler"] = sampler_name
        metadata["Scheduler"] = scheduler
        metadata["Denoise"] = denoise

        # Use the separate flux_guidance float for CFG scale
        metadata["CFG scale"] = flux_guidance

        used_ckpt = str(unet_name).strip() if unet_name and str(unet_name).strip() else "UnknownModel"
        model_filename, model_short_hash = find_model_file_and_hash(used_ckpt)
        metadata["Model"] = unet_name.strip() if unet_name.strip() else "UnknownModel"
        metadata["Model hash"] = unet_hash if unet_hash else "no_hash"

        metadata["Positive prompt"] = prompt_text
        metadata["Negative prompt"] = ""

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
                            hash_dict["lora:" + base] = short_sha256(lora_path)
                        else:
                            hash_dict["lora:" + base] = base
        metadata["Hashes"] = json.dumps(hash_dict)
        metadata_str = json.dumps(metadata, indent=2)

        return ({"samples": out["samples"]}, metadata, metadata_str)

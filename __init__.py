from .MangoTriggerExporter import MangoTriggerExporter
from .KSamplerMango import KSamplerMango
from .MangoLoader import MangoLoader
from .ImageSaverMango import ImageSaverMango
from .LatentImageMango import LatentImageMango
from .PromptMango import PromptMango
from .PromptEmbedMango import PromptEmbedMango
from .CompositeMangoLoader import CompositeMangoLoader
from .FluxSamplerMango import FluxSamplerMango
from .FluxGuidanceMango import FluxGuidanceMango
from .MangoPromptSave import PromptSave
from .MangoPromptLoad import MangoPromptLoad
from .LoraStackMango import LoraStackMango
from .MangoImageLoader import MangoImageLoader
from .MangoLoader10Loras import MangoLoader10Loras
from .MangoModelData import MangoModelData

NODE_CLASS_MAPPINGS = {
    "MangoTriggerExporter":     MangoTriggerExporter,
    "KSamplerMango":            KSamplerMango,
    "MangoLoader":              MangoLoader,
    "ImageSaverMango":          ImageSaverMango,
    "LatentImageMango":         LatentImageMango,
    "PromptMango":              PromptMango,
    "PromptEmbedMango":         PromptEmbedMango,
    "CompositeMangoLoader":     CompositeMangoLoader,
    "FluxSamplerMango":         FluxSamplerMango,
    "FluxGuidanceMango":        FluxGuidanceMango,
    "PromptSave":               PromptSave,
    "MangoPromptLoad":          MangoPromptLoad,
    "LoraStackMango":           LoraStackMango,
    "MangoImageLoader":         MangoImageLoader,
    "MangoLoader10Loras":       MangoLoader10Loras,
    "MangoModelData":           MangoModelData,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangoTriggerExporter":      "Trigger Exporter (Mango)",
    "KSamplerMango":             "KSampler (Mango)",
    "MangoLoader":               "Loader (Mango)",
    "ImageSaverMango":           "Image Saver (Mango)",
    "LatentImageMango":          "Latent Image (Mango)",
    "PromptMango":               "Prompt (Mango)",
    "PromptEmbedMango":          "Prompt /w Embedding (Mango)",
    "CompositeMangoLoader":      "Diffusion Loader (Mango)",
    "FluxSamplerMango":          "FluxSampler (Mango)",
    "FluxGuidanceMango":         "FluxGuidance (Mango)",
    "PromptSave":                "Save Prompt (Mango)",
    "MangoPromptLoad":           "Load Prompt (Mango)",
    "LoraStackMango":            "LoRA Stack (Mango)",
    "MangoImageLoader":          "Image Loader (Mango)",
    "MangoLoader10Loras":        "Loader (Mango + 10 Loras)",
    "MangoModelData":            "Model Data (Mango)",
}

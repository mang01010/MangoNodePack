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

NODE_CLASS_MAPPINGS = {
    "MangoTriggerExporter": MangoTriggerExporter,
    "KSamplerMango":       KSamplerMango,
    "MangoLoader":         MangoLoader,
    "ImageSaverMango":     ImageSaverMango,
    "LatentImageMango":    LatentImageMango,
    "PromptMango":         PromptMango,
    "PromptEmbedMango":    PromptEmbedMango,
    "CompositeMangoLoader": CompositeMangoLoader,
    "FluxSamplerMango":    FluxSamplerMango,
    "FluxGuidanceMango":   FluxGuidanceMango,
    "PromptSave":          PromptSave,
    "MangoPromptLoad":     MangoPromptLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangoTriggerExporter": "Trigger Exporter (Mango)",
    "KSamplerMango":       "KSampler (Mango)",
    "MangoLoader":         "Loader (Mango)",
    "ImageSaverMango":     "Image Saver (Mango)",
    "LatentImageMango":    "Latent Image (Mango)",
    "PromptMango":         "Prompt (Mango)",
    "PromptEmbedMango":    "Prompt /w Embedding (Mango)",
    "CompositeMangoLoader": "Diffusion Loader (Mango)",
    "FluxSamplerMango":    "FluxSampler (Mango)",
    "FluxGuidanceMango":   "FluxGuidance (Mango)",
    "PromptSave":          "Save Prompt (Mango)",
    "MangoPromptLoad":     "Load Prompt (Mango)",
}

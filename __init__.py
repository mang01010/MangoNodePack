from .MangoTriggerExporter import MangoTriggerExporter
from .KSamplerMango import KSamplerMango
from .MangoLoader import MangoLoader
from .ImageSaverMango import ImageSaverMango
from .LatentImageMango import LatentImageMango
from .PromptMango import PromptMango

NODE_CLASS_MAPPINGS = {
    "MangoTriggerExporter": MangoTriggerExporter,
    "KSamplerMango": KSamplerMango,
    "MangoLoader": MangoLoader,
    "ImageSaverMango": ImageSaverMango,
    "LatentImageMango": LatentImageMango,
    "PromptMango": PromptMango
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangoTriggerExporter": "Trigger Exporter (Mango)",
    "KSamplerMango": "KSampler (Mango)",
    "MangoLoader": "Loader (Mango)",
    "ImageSaverMango": "Image Saver (Mango)",
    "LatentImageMango": "Latent Image (Mango)",
    "PromptMango": "Prompt (Mango)"
}

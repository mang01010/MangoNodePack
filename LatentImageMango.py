import torch

class LatentImageMango:
    @classmethod
    def INPUT_TYPES(cls):
        dimensions = [
            "1:1 square 1024x1024",
            "3:4 portrait 896x1152",
            "5:8 portrait 832x1216",
            "9:16 portrait 768x1344",
            "9:21 portrait 640x1536",
            "4:3 landscape 1152x896",
            "3:2 landscape 1216x832",
            "16:9 landscape 1344x768",
            "21:9 landscape 1536x640",
            "32:9 landscape 1600x450"
        ]
        return {
            "required": {
                "dimensions": (dimensions,)
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent_image", "width", "height")
    FUNCTION = "compute_dimensions"
    CATEGORY = "Mango Node Pack/Loaders"

    def compute_dimensions(self, dimensions):
        if dimensions == "1:1 square 1024x1024":
            width, height = 1024, 1024
        elif dimensions == "3:4 portrait 896x1152":
            width, height = 896, 1152
        elif dimensions == "5:8 portrait 832x1216":
            width, height = 832, 1216
        elif dimensions == "9:16 portrait 768x1344":
            width, height = 768, 1344
        elif dimensions == "9:21 portrait 640x1536":
            width, height = 640, 1536
        elif dimensions == "4:3 landscape 1152x896":
            width, height = 1152, 896
        elif dimensions == "3:2 landscape 1216x832":
            width, height = 1216, 832
        elif dimensions == "16:9 landscape 1344x768":
            width, height = 1344, 768
        elif dimensions == "21:9 landscape 1536x640":
            width, height = 1536, 640
        elif dimensions == "32:9 landscape 1600x450":
            width, height = 1600, 450
        else:
            width, height = 1024, 1024
        latent_image = torch.zeros([1, 4, height // 8, width // 8])
        return {"samples": latent_image}, width, height


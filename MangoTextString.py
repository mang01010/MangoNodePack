TEXT_TYPE = "STRING"  # ComfyUI string type

class MangoTextString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"face": ("STRING", {"default": '', "multiline": True})},
            "optional": {
                "eyes": ("STRING", {"default": '', "multiline": True}),
                "hands": ("STRING", {"default": '', "multiline": True}),
                "feet": ("STRING", {"default": '', "multiline": True}),
                "shoes": ("STRING", {"default": '', "multiline": True}),
                "gen1": ("STRING", {"default": '', "multiline": True}),
                "gen2": ("STRING", {"default": '', "multiline": True}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,) * 7  # 7 outputs
    RETURN_NAMES = ("face", "eyes", "hands", "feet", "shoes", "gen1", "gen2")
    FUNCTION = "text_string"
    CATEGORY = "Mango Node Pack/Metadata"

    def text_string(self, face='', eyes='', hands='', feet='', shoes='', gen1='', gen2=''):
        # just return all inputs as-is
        return (face, eyes, hands, feet, shoes, gen1, gen2)

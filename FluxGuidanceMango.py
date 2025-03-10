import node_helpers

class FluxGuidanceMango:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("conditioned", "flux_guidance")
    FUNCTION = "append"
    CATEGORY = "Mango Node Pack/Metadata"

    def append(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, guidance)

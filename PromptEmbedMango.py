class PromptEmbedMango:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "QualityTags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "rows": 3
                    }
                ),
                "SceneDescription": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "rows": 3
                    }
                ),
                "Tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "rows": 3
                    }
                ),
                "Embeddings": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "rows": 3
                    }
                ),
            },
            "optional": {
                "string": (
                    "STRING",
                    {
                        "default": ""
                    }
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Mango Node Pack/Metadata"

    def execute(self, QualityTags, SceneDescription, Tags, Embeddings, string=""):
        parts = [s.strip() for s in (QualityTags, SceneDescription, Tags, Embeddings, string) if s.strip()]
        concatenated = ", ".join(parts)
        return (concatenated,)

class PromptMango:

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
                "string": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                    }
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "Mango Node Pack/Metadata"

    def execute(self, QualityTags, SceneDescription, Tags, string):
        parts = [s.strip() for s in (QualityTags, SceneDescription, Tags, string) if s.strip()]
        concatenated = ", ".join(parts)
        return (concatenated,)

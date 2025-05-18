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
                "CharacterTags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "rows": 3
                    }
                ),
                "SceneTags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "rows": 3
                    }
                ),
                "BackgroundTags": (
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

    def execute(self, QualityTags="", SceneDescription="", CharacterTags="", SceneTags="", BackgroundTags="", Embeddings="", string=None):

        if string is None:
            string = ""

        parts = [s.strip() for s in (QualityTags, SceneDescription, CharacterTags, SceneTags, BackgroundTags, Embeddings, string) if s.strip()]
        concatenated = ", ".join(parts)

        return (concatenated,)

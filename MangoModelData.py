import os
import json

class MangoModelData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Checkpoint name passed from KSamplerMango (may include subfolder)
                "ckpt_name": ("STRING", {"tooltip": "Checkpoint name"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "export_model_data"
    CATEGORY = "Mango Node Pack/Metadata"

    def export_model_data(self, ckpt_name):
        # Determine path to JSON file containing manual model usage tips
        data_path = os.path.join(os.path.dirname(__file__), 'model_data.json')

        # Check for existence
        if not os.path.isfile(data_path):
            return (f"[ModelData] ERROR: file not found: {data_path}",)

        # Load JSON
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                db = json.load(f)
        except Exception as e:
            return (f"[ModelData] ERROR: failed to load JSON ({type(e).__name__}): {e}",)

        # Validate JSON structure
        if not isinstance(db, dict) or not db:
            return ("[ModelData] ERROR: model_data.json is empty or not a JSON object",)

        # Extract the filename portion
        base_name = os.path.basename(ckpt_name)

        # Lookup strategy: full key, basename, then case-insensitive basename
        model_entry = db.get(ckpt_name)
        if model_entry is None:
            model_entry = db.get(base_name)
        if model_entry is None:
            lower_base = base_name.lower()
            for key, val in db.items():
                if os.path.basename(key).lower() == lower_base:
                    model_entry = val
                    break

        # If still not found, list available keys in the error
        if model_entry is None:
            available = ", ".join(db.keys())
            return (f"[ModelData] no entry for '{base_name}'. Available keys: {available}",)

        # Format and return the JSON snippet using the base filename
        snippet = json.dumps({base_name: model_entry}, indent=2)
        return (snippet,)

"""
Dependencies:
- pip install safetensors
- ComfyUI's 'folder_paths' for locating LoRA files.
"""

import os
import hashlib
import json
import requests

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None

import folder_paths

#METADATA FETCHING

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def get_model_version_info(hash_value):
    api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        return {}

def parse_local_safetensors_metadata(lora_path):
    if not safe_open:
        return {}
    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            print(f"DEBUG: Safetensors metadata for {lora_path} -> {meta}")
            return meta
    except Exception as e:
        print(f"Failed reading local safetensors metadata for {lora_path}: {e}")
        return {}

def extract_trigger_words_from_metadata(meta):
    triggers_found = set()

    if "trainedWords" in meta and meta["trainedWords"]:
        val = meta["trainedWords"]
        if isinstance(val, str):
            triggers_found.update(x.strip() for x in val.split(",") if x.strip())
        elif isinstance(val, list):
            triggers_found.update(val)
        elif isinstance(val, dict):
            triggers_found.update(val.keys())

    if "modelspec.trigger_phrase" in meta:
        trigger_phrase = meta["modelspec.trigger_phrase"]
        if isinstance(trigger_phrase, str) and trigger_phrase.strip():
            triggers_found.add(trigger_phrase.strip())

    return list(triggers_found)

def get_lora_metadata(lora_name):
    db_path = os.path.join(os.path.dirname(__file__), 'lora_metadata_db.json')
    try:
        with open(db_path, 'r') as f:
            db = json.load(f)
    except Exception:
        db = {}
    if lora_name in db:
        return db[lora_name]
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not os.path.exists(lora_path):
        db[lora_name] = {"triggerWords": ""}
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=4)
        return db[lora_name]
    meta = parse_local_safetensors_metadata(lora_path)
    local_triggers = extract_trigger_words_from_metadata(meta)
    if not local_triggers:
        LORAsha256 = calculate_sha256(lora_path)
        model_info = get_model_version_info(LORAsha256)
        if model_info.get("trainedWords"):
            local_triggers = model_info["trainedWords"]
    triggers_str = ", ".join(local_triggers)
    db[lora_name] = {"triggerWords": triggers_str}
    try:
        with open(db_path, 'w') as f:
            json.dump(db, f, indent=4)
    except Exception as e:
        print(f"Error saving metadata cache: {e}")
    return db[lora_name]

#NODE

class MangoTriggerExporter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "export_triggerwords"
    CATEGORY = "Mango Node Pack/Metadata"

    def export_triggerwords(self, lora_stack):
        triggerwords_list = []
        for item in lora_stack:
            lora_name = item[0] if isinstance(item, (tuple, list)) and item else None
            if lora_name and lora_name != "None":
                try:
                    metadata = get_lora_metadata(lora_name)
                    tw = metadata.get("triggerWords", "")
                    if tw:
                        triggerwords_list.append(tw)
                except Exception as e:
                    print(f"Error processing '{lora_name}': {e}")
        combined_triggerwords = ", ".join(triggerwords_list)
        return (combined_triggerwords,)

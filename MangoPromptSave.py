import os
import re
from datetime import datetime
import folder_paths

class PromptSave:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "The prompt text to save."}),
                "filename_prefix": ("STRING", {"default": "Prompt", "tooltip": "Base name for the saved prompt file."}),
                "subdirectory_name": ("STRING", {"default": "Prompts", "tooltip": "Subfolder (inside the output folder) to save the prompt."}),
            }
        }
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "save_prompt"
    CATEGORY = "Mango Node Pack/Metadata"
    DESCRIPTION = "Saves prompt text to a .txt file with a simple incrementing suffix."

    def save_prompt(self, prompt, filename_prefix="Prompt", subdirectory_name="Prompts"):
        try:
            # Get the output directory from folder_paths.
            output_dir = folder_paths.get_output_directory()
            if not output_dir:
                print("Error: Output directory not found!")
                return ("",)

            # Determine the target folder.
            full_output_folder = os.path.join(output_dir, subdirectory_name) if subdirectory_name else output_dir
            os.makedirs(full_output_folder, exist_ok=True)

            # Find the next available file number.
            pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d+)\.txt$")
            max_num = 0
            for f in os.listdir(full_output_folder):
                m = pattern.match(f)
                if m:
                    try:
                        num = int(m.group(1))
                        if num > max_num:
                            max_num = num
                    except Exception as e:
                        print("Error parsing file number in file", f, ":", e)
            next_num = max_num + 1

            file_name = f"{filename_prefix}_{next_num}.txt"
            full_path = os.path.join(full_output_folder, file_name)
            print("Saving prompt to:", full_path)

            # Save the prompt using UTF-8 encoding.
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(prompt)

            return (prompt,)
        except Exception as e:
            print("Error saving prompt:", e)
            return (prompt,)

import os
import folder_paths

class MangoPromptLoad:
    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        choices = []
        if output_dir and os.path.exists(output_dir):
            # Recursively search for .txt files in output_dir and subdirectories
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(".txt"):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        choices.append(rel_path)
        choices = sorted(choices)
        # Provide a fallback if no files found
        if not choices:
            choices = [""]
        return {
            "required": {
                "filename": (choices, {"tooltip": "Select a .txt file from the output folder and its subfolders."})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_prompt"
    CATEGORY = "Mango Node Pack/Metadata"
    DESCRIPTION = (
        "Loads prompt text from a .txt file found in the default ComfyUI output folder "
        "and all its subfolders. Perfect to serve up a prompt for PromptMango!"
    )

    def load_prompt(self, filename):
        output_dir = folder_paths.get_output_directory()
        if not output_dir:
            print("Error: Output directory not found! Check your ComfyUI setup.")
            return ("",)

        full_path = os.path.join(output_dir, filename)
        if not os.path.exists(full_path):
            print(f"Error: File {full_path} not found.")
            return ("",)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
            return (text,)
        except Exception as e:
            print("Error loading file:", e)
            return ("",)

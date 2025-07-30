import os
import json
import re

# === Paths ===
base_dir = '/home/disk/CR2C2/test_data_organized'
captions_dir = os.path.join(base_dir, '4o_evaluation_formatted_cleaned')
videos_dir = os.path.join(base_dir, 'videos')
prompt_path = os.path.join(base_dir, 'test_prompt_template.txt')
output_json = os.path.join(base_dir, '4o_caption_test_template.json')

# === Load the full unified prompt ===
with open(prompt_path, 'r', encoding='utf-8') as f:
    full_prompt = f.read().strip()

# === Unicode cleanup function ===
def clean_unicode(text):
    text = text.replace('\u2003', ' ')           # Replace em space
    text = re.sub(r'\n+', '\n', text)            # Collapse multiple \n to one
    return text.strip()

# === Build entries ===
all_data = []

for filename in os.listdir(captions_dir):
    if not filename.endswith('.txt'):
        continue
    file_base = filename.replace("_ground_truth.txt", "")
    video_filename = f"{file_base}.mp4"
    abs_video_path = os.path.abspath(os.path.join(videos_dir, video_filename))

    # Load caption as full response
    with open(os.path.join(captions_dir, filename), 'r', encoding='utf-8') as f:
        full_response = f.read().strip()

    # Clean prompt and caption
    cleaned_prompt = clean_unicode(full_prompt)
    cleaned_response = clean_unicode(full_response)

    # Format for single-turn conversation
    entry = {
        "video": abs_video_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n{cleaned_prompt}"
            },
            {
                "from": "gpt",
                "value": cleaned_response
            }
        ]
    }

    all_data.append(entry)

# === Save output ===
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"✅ Created {len(all_data)} entries in {output_json} using unified prompt format.")

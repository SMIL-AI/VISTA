import os
import json
import re

# === Paths ===
base_dir = '/home/disk/CR2C2/extracted_data'
captions_dir = os.path.join(base_dir, 'formatted_caption_cleaned')
videos_dir = os.path.join(base_dir, 'videos')
output_json = os.path.join(base_dir, 'train_weather_qwen_video_with_template.json')

# === Unified prompt template ===
template_prompt = """
Based on the input video, perform a structured scene analysis and risk assessment.
Please fill in the following template exactly, replacing the placeholders with your observations.
Stick to bullet points and short complete sentences where applicable.

**Structured Scene Analysis**
- **Time of Day**: {time_of_day}
- **Weather Conditions**: {weather}
- **Pavement Wetness**: {wetness}
- **Vehicle Behavior**: {vehicle_behavior}
- **Traffic Flow & Speed**: {traffic_flow}
- **Congestion Level**: {congestion}

**Summary**  
{summary}

**Risk Report**
1. **Environmental Risk**: {env_risk}
2. **Vehicle Behavior Risk**: {veh_risk}
3. **Traffic Flow Risk**: {flow_risk}
4. **Overall Safety Risk Level**: {risk_level}
5. **Driver Alerts**: {driver_alerts}
6. **Suggested Safe Speed**: {safe_speed}
""".strip()

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

    file_base = filename.replace("_risk_analysis.txt", "")
    # Try matching .mp4 or .ts based on availability
    mp4_path = os.path.join(videos_dir, f"{file_base}.mp4")
    ts_path = os.path.join(videos_dir, f"{file_base}.ts")
    if os.path.exists(mp4_path):
        abs_video_path = os.path.abspath(mp4_path)
    elif os.path.exists(ts_path):
        abs_video_path = os.path.abspath(ts_path)
    else:
        print(f"[WARNING] No video file found for base name: {file_base}")
        continue

    # Load full response
    with open(os.path.join(captions_dir, filename), 'r', encoding='utf-8') as f:
        full_response = f.read().strip()

    # Try to split scene and risk parts
    if "Risk Report" in full_response:
        scene_analysis, risk_report = full_response.split("Risk Report", 1)
        risk_report = "Risk Report" + risk_report
    else:
        lines = full_response.splitlines()
        midpoint = len(lines) // 2
        scene_analysis = "\n".join(lines[:midpoint]).strip()
        risk_report = "\n".join(lines[midpoint:]).strip()

    # Clean up
    scene_analysis = clean_unicode(scene_analysis)
    risk_report = clean_unicode(risk_report)
    full_cleaned_answer = f"{scene_analysis}\n\n{risk_report}".strip()
    prompt_text = clean_unicode(template_prompt)

    # Add to dataset
    entry = {
        "video": abs_video_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n{prompt_text}"
            },
            {
                "from": "gpt",
                "value": full_cleaned_answer
            }
        ]
    }

    all_data.append(entry)

# === Save output ===
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"✅ Created {len(all_data)} entries in {output_json} using unified prompt format.")

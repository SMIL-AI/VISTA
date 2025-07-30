import json
import os
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime
from utils import compute_metrics_single  

def load_model_robust(model_path):
    print(f"🚀 Loading model from: {model_path}")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True
    )
    print("✅ Model loaded successfully!")
    return model, processor, device

def get_video_files(video_dir, limit=None):
    video_extensions = {'.mp4'}
    video_files = [
        os.path.join(video_dir, file)
        for file in os.listdir(video_dir)
        if any(file.lower().endswith(ext) for ext in video_extensions)
    ]
    if limit:
        video_files = video_files[:limit]
    video_files = sorted(video_files, key=lambda x: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', x)])
    print(f"📹 Found {len(video_files)} video files")
    return video_files

def predict_on_video_robust(model, processor, device, video_path, question):
    try:
        print(f"Processing: {os.path.basename(video_path)}")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        except:
            image_inputs, video_inputs = process_vision_info(messages)
            video_kwargs = {}

        for i, v in enumerate(video_inputs):
            if isinstance(v, torch.Tensor):
                video_inputs[i] = v.to(device)
        for k, v in video_kwargs.items():
            if isinstance(v, torch.Tensor):
                video_kwargs[k] = v.to(device)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )

        processed_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor)
                else [item.to(device) for item in v] if isinstance(v, list) and all(isinstance(item, torch.Tensor) for item in v)
                else v)
            for k, v in inputs.items()
        }

        if hasattr(model.config, 'vision_config') and hasattr(model.config.vision_config, 'tokens_per_second'):
            if isinstance(model.config.vision_config.tokens_per_second, torch.Tensor):
                model.config.vision_config.tokens_per_second = model.config.vision_config.tokens_per_second.to(device)

        with torch.no_grad():
            generated_ids = model.generate(**processed_inputs, max_new_tokens=512)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(processed_inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del image_inputs, video_inputs, video_kwargs, inputs, processed_inputs
        del generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_text.strip()
    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        return f"ERROR: {str(e)}"

def compute_avg(metrics_list):
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()} if metrics_list else {}


# def load_ground_truth(gt_path):
#     with open(gt_path, 'r', encoding='utf-8') as f:
#         gt_data = json.load(f)
    
#     # Create a mapping from video filename to cleaned caption
#     gt_map = {
#         os.path.basename(item["video"]): item["cleaned_caption"]
#         for item in gt_data
#     }
    
#     return gt_map
def load_ground_truth(gt_path):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    gt_map = {}

    for item in gt_data:
        video_name = os.path.basename(item["video"])
        # Find the GPT reply in conversations
        gpt_reply = next((conv["value"] for conv in item["conversations"] if conv["from"] == "gpt"), None)

        if gpt_reply:
            # Remove explanation or wrapping if needed (e.g., remove leading junk)
            if "**Structured Scene Analysis**" in gpt_reply:
                gpt_reply = gpt_reply[gpt_reply.find("**Structured Scene Analysis**"):]
            gt_map[video_name] = gpt_reply.strip()

    return gt_map


def main():
    model_path = "/home/disk/ning/vLLM/Qwen2.5-VL/qwen-vl-finetune/checkpoints/transgpt_3b_mlp_template"
    test_prompt_path = '/home/disk/CR2C2/test_data_organized/test_prompt_template.txt'
    video_dir = "/home/disk/CR2C2/test_data_organized/videos"
    output_file = "test_video_predictions_with_metrics_3B_mlp_template.json"
    gt_caption_path = "/home/disk/CR2C2/test_data_organized/4o_caption_test_template.json"

    print("🎬 Starting Robust Video Prediction + Evaluation")
    print("=" * 70)
    print(f"📂 Model: {model_path}")
    print(f"📹 Videos: {video_dir}")
    print(f"💾 Output: {output_file}")
    print("=" * 70)

    try:
        model, processor, device = load_model_robust(model_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    video_files = get_video_files(video_dir)
    if not video_files:
        print("❌ No video files found!")
        return

    with open(test_prompt_path, 'r', encoding='utf-8') as f:
        question = f.read().strip()

    gt_map = load_ground_truth(gt_caption_path)
    results = []

    for video_file in video_files:
        video_name = os.path.basename(video_file)
        if any(r.get("video_name") == video_name for r in results):
            print(f"Skipping {video_name} (already processed)")
            continue

        prediction = predict_on_video_robust(model, processor, device, video_file, question)
        success = not prediction.startswith("ERROR")

        # Evaluation Metrics
        metrics = {}
        if success and video_name in gt_map:
            gt_caption = gt_map[video_name]
            metrics = compute_metrics_single(prediction, gt_caption)

        result = {
            "video_name": video_name,
            "video_path": video_file,
            "question": question,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "metrics": metrics
        }

        results.append(result)

        if success:
            print(f"✅ {video_name} — Prediction: {prediction[:100]}{'...' if len(prediction) > 100 else ''}")
        else:
            print(f"❌ {video_name} — Error: {prediction}")

        print(f"📈 Metrics: {metrics}")
        print("-" * 50)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print("\n📊 SUMMARY:")
    print(f"✅ Successful predictions: {successful}")
    print(f"❌ Failed predictions: {failed}")
    print(f"💾 Output saved to: {output_file}")

    successful_metrics = [r["metrics"] for r in results if r["success"] and r["metrics"]]
    avg_metrics = compute_avg(successful_metrics)

    if avg_metrics:
        print("\n📈 AVERAGE METRICS:")
        for k, v in avg_metrics.items():
            print(f"  {k.upper():<8}: {v:.4f}")
    else:
        print("⚠️ No metrics computed — check data and predictions.")

    if successful > 0:
        print("\n📝 Sample successful predictions with metrics:")
        for r in results:
            if r["success"]:
                print(f"🎬 {r['video_name']}")
                print(f"🤖 Prediction: {r['prediction'][:150]}...")
                print(f"📊 Metrics: {r['metrics']}")
                print()
                break  # Show only one sample

if __name__ == "__main__":
    main()

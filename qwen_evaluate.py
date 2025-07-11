import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
import re

# ---- Load model and processor ----
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# ---- Load anomaly summary CSV ----
csv_path = r"D:/OneDrive/Documents/vidresearch/anomaly_summary.csv"
df = pd.read_csv(csv_path)

# ---- Text cleaning utility ----
def normalize(text):
    return re.sub(r"[^a-z0-9]", " ", text.lower()).strip()

# ---- Set video directory ----
video_dir = Path(r"D:/OneDrive/Documents/vidresearch/output_videos")

# ---- Evaluation results ----
results = []

# ---- Main loop: Analyze each video ----
for _, row in df.iterrows():
    video_name = row['video_name']
    video_path = video_dir / video_name

    print(f"🔍 Processing: {video_name}")

    # ---- Prepare chat input ----
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": str(video_path),
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video in detail."},
        ],
    }]

    # ---- Apply template and extract inputs ----
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # ---- Generate output ----
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    # ---- Decode output ----
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    # ---- Normalize output ----
    generated_clean = normalize(output_text)
    generated_words = set(generated_clean.split())

    # ---- Ground truth comparison ----
    activity_clean = normalize(row['original_activity'])
    activity_found = activity_clean in generated_clean

    anomalies = row['anomalies'].split(";")
    anomalies_clean = [
        normalize(a.split("_")[-1]) if "combined_activity" in a else normalize(a)
        for a in anomalies
    ]

    detected = []
    missed = []

    for anomaly, clean in zip(anomalies, anomalies_clean):
        if clean in generated_words or clean in generated_clean:
            detected.append(anomaly)
        else:
            missed.append(anomaly)

    # ---- Scoring ----
    score = int(activity_found) + len(detected)
    total_possible = 1 + len(anomalies)
    accuracy = score / total_possible

    # ---- Append results ----
    results.append({
        "video_name": video_name,
        "original_activity": row['original_activity'],
        "expected_anomalies": anomalies,
        "generated_description": output_text,
        "activity_match": activity_found,
        "detected_anomalies": detected,
        "missed_anomalies": missed,
        "score": f"{score}/{total_possible}",
        "accuracy": round(accuracy, 2),
    })

# ---- Save results ----
out_df = pd.DataFrame(results)
out_df.to_csv("anomaly_evaluation_results.csv", index=False)

print("\n✅ Evaluation complete.")
print("📄 Results saved to: anomaly_evaluation_results.csv")

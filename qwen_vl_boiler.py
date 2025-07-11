import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load the model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# ---- Configurable video path ----
video_path = r"D:/OneDrive/Documents/vidresearch/output_videos/anomaly_video_0.mp4"  # Windows: file:///D:/full/path/video.mp4

# ---- Prompt as message format ----
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 360 * 420,  # optional: reduce resolution
                "fps": 1.0,  # optional: control frame sampling rate
            },
            {"type": "text", "text": "Describe this video in detail."},
        ],
    }
]

# ---- Apply chat template ----
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ---- Extract visual inputs ----
image_inputs, video_inputs = process_vision_info(messages)

# ---- Final model input ----
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# ---- Inference ----
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256)

# ---- Post-process output ----
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# ---- Display output ----
print("\n📹 Video Description:\n", output_text[0])

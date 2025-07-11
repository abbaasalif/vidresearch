# 🎥 VidResearch: Synthetic Video Anomalies + LLM-Based Evaluation

This project focuses on generating **synthetic anomalies from the UCF101 dataset** and evaluating whether large vision-language models like **Qwen2.5-VL** can correctly describe and detect those anomalies. It combines classical computer vision techniques with cutting-edge multimodal AI for video understanding.

---

## 🔍 Overview

* ✅ Generates synthetic anomalies from real human activity videos (UCF101).
* 🎭 Anomalies include: **frame reversal**, **temporal shuffling**, **object insertion**, and **activity blending**.
* 🤖 Uses **Qwen2.5-VL**, a vision-language LLM, to caption anomaly videos.
* 🧪 Evaluates LLM descriptions vs. ground truth anomalies.
* 📊 Outputs a detailed CSV report with precision and detection scores.

---

## 📁 Repository Structure

```
vidresearch/
├── create_anomaly.py              # Generate synthetic anomaly videos
├── qwen_evaluate.py               # Evaluate anomaly detection via Qwen2.5-VL
├── anomaly_summary.csv            # Metadata for generated videos
├── anomaly_evaluation_results.csv # Evaluation output after LLM analysis
├── output_videos/                 # Contains generated .mp4 anomaly videos
├── synthetic_objects/             # Random object overlays (RGBA)
└── UCF-101/                       # Place extracted UCF-101 dataset here
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/abbaasalif/vidresearch.git
cd vidresearch
```

### 2. Install dependencies

> ✅ Python ≥ 3.8 required

```bash
pip install torch pandas opencv-python transformers
```

Make sure your machine has a GPU (NVIDIA recommended) with CUDA support for Qwen2.5-VL inference.

---

## 🎜️ Step 1: Download UCF101 Dataset

1. Visit the [UCF101 download page](https://www.crcv.ucf.edu/data/UCF101.php)
2. Register and download the dataset (Split 1, .avi format)
3. Extract it into the project folder as:

```
vidresearch/
└── UCF-101/
    ├── ApplyEyeMakeup/
    │   ├── video1.avi
    │   └── ...
    ├── Basketball/
    └── ...
```

---

## ⚙️ Step 2: Generate Anomaly Videos

Run the script below to generate 100 synthetic anomaly videos:

```bash
python create_anomaly.py
```

Each output video will:

* Combine two real videos from different activities
* Optionally apply frame reversal, shuffling, and object insertion
* Save metadata in `anomaly_summary.csv`

Output location: `output_videos/anomaly_video_0.mp4` to `anomaly_video_99.mp4`

---

## 🤖 Step 3: Evaluate Anomalies with Qwen2.5-VL

This script uses [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) to generate captions for the anomaly videos, and compares the description against known injected anomalies.

```bash
python qwen_evaluate.py
```

> ⚠️ Modify paths in `qwen_evaluate.py` (`csv_path` and `video_dir`) to match your machine.

This will:

* Generate natural language descriptions for each video
* Detect whether anomalies and original activity are correctly described
* Save output to: `anomaly_evaluation_results.csv`

---

## 📊 Output Example

| video\_name           | original\_activity | detected\_anomalies               | score | accuracy |
| --------------------- | ------------------ | --------------------------------- | ----- | -------- |
| anomaly\_video\_0.mp4 | Basketball         | \['shuffled', 'inserted\_object'] | 3/4   | 0.75     |
| anomaly\_video\_1.mp4 | Running            | \['reversed']                     | 2/3   | 0.67     |

---

## 💡 Applications

* Benchmarking LLMs on multimodal understanding
* Generating training data for anomaly detection in videos
* Simulating rare or dangerous scenarios for surveillance datasets

---

## 🧠 Model Used

* **Qwen2.5-VL-7B-Instruct**: A multimodal large language model from Alibaba Cloud

  * Repo: [Qwen2.5-VL on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  * Can handle both video and text inputs
  * Used here for zero-shot anomaly captioning and reasoning

---

## 📄 License

MIT License © 2024 Abbaas Alif Mohamed Nishar
Use this repo freely for academic or research purposes. Please cite or link the repo if you use this work in publications.

---

## 🙌 Acknowledgements

* [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
* [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) by Alibaba Cloud
* [Transformers by HuggingFace](https://github.com/huggingface/transformers)

---

## ✨ Contact

Questions or suggestions?
Feel free to open an issue or contact [@abbaasalif](https://github.com/abbaasalif)

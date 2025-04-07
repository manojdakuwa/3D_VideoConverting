# 🧠 3D Video Generator using Depth Estimation

This project generates 3D videos from 2D input videos by performing **depth estimation** and **stereo image generation** using deep learning models like MiDaS and Transformer-based architectures.

---

## 📽️ Features

- Extract frames from an input video
- Estimate depth maps using a pre-trained MiDaS model
- Generate stereo left-right views to simulate 3D perspective
- Combine stereo frames into a final 3D video
- Output in `.mp4` format

---

## 🛠️ Tech Stack

- Python 3.10+
- OpenCV
- PyTorch
- TensorFlow
- Transformers (HuggingFace)
- NumPy

---

## 📦 Setup

### 1. Clone the Repository
```bash
git clone https://github.com/manojdakuwa/3D_VideoConverting.git
cd 3d-video-generator
python -m venv depth_env
source depth_env/bin/activate  # On Windows: depth_env\Scripts\activate
python -m venv depth_env
source depth_env/bin/activate  # On Windows: depth_env\Scripts\activate
pip install -r requirements.txt

3d-video-generator/
│
├── data/                     # Input videos
├── output/                   # Output 3D videos
├── scripts/
│   ├── video_processing.py   # Frame extraction
│   ├── depth_estimation.py   # MiDaS model loading and prediction
│   ├── stereo_conversion.py  # Stereo image generation
│   └── video_output.py       # Output 3D video writer
│
├── main.py                   # Main pipeline
├── requirements.txt
├── .gitignore
└── README.md

python main.py


⚠️ Notes
Make sure numpy version is compatible with TensorFlow (<2.2.0)

You may need to adjust frame resolution or video codec based on your system

Tested with MiDaS (Intel/dpt-hybrid-midas) from Hugging Face


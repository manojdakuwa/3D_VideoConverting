import transformers
from scripts.video_processing import extract_frames
from scripts.depth_estimation import estimate_depth
from scripts.stereo_conversion import create_stereo_frames
from scripts.video_output import generate_3d_video

import cv2
import numpy as np
import torch
import tensorflow as tf

print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("Torch version:", torch.__version__)
print("TensorFlow version:", tf.__version__)
print("transformer version:", transformers.__version__)

# Load video
video_path = "data/video2.mp4"
frames = extract_frames(video_path)

# Apply depth estimation
depth_frames = [estimate_depth(frame) for frame in frames]

# Create stereo views
# stereo_frames = [create_stereo_frames(frame, depth) for frame, depth in zip(frames, depth_frames)]
# combined_frames = [np.hstack((left, right)) for left, right in stereo_frames]

# Create stereo views
stereo_frames = [create_stereo_frames(frame, depth) for frame, depth in zip(frames, depth_frames)]

# Combine left and right views into one frame (side-by-side)
combined_frames = [np.hstack((left, right)) for left, right in stereo_frames]

# Generate final 3D video
generate_3d_video(combined_frames, "output/3d_video.mp4")

print("✅ 3D video successfully generated!")
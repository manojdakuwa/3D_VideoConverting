import torch
import cv2
import numpy as np
from huggingface_hub import login
from transformers import DPTFeatureExtractor, DPTForDepthEstimation, AutoModelForDepthEstimation, AutoFeatureExtractor
#print(transformers.__version__)

login(token="hf_XXXXXX")  # Replace with your Hugging Face token
# Load pre-trained model
model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", cache_dir="./models")
feature_extractor = AutoFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas", cache_dir="./models")

def estimate_depth(image):
    """Estimate depth from a single frame."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    
    return depth.squeeze().cpu().numpy()
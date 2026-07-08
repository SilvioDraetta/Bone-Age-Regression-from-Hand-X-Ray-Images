import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

import torch
from .load_birefnet import load_birefnet
from src.preprocessing.segmentation import extract_object
import json
from .loader import load_bone_age_model
from src.preprocessing.transforms import transform

with open("src/config/stats.json") as f:
    stats = json.load(f)

MEAN = stats["mean"]
STD = stats["std"]

def run_bone_age_inference(image_path, sex_bool, device="cuda"):
    
    model_biref = load_birefnet()
    image, _ = extract_object(model_biref, image_path)

    image = image.convert("L")  

    # Convert to tensor
    img_tensor = transform(image).unsqueeze(0).to(device)
    male_tensor = torch.tensor([[1.0 if sex_bool else 0.0]], device=device)

    model = load_bone_age_model("notebook/model_results/torch_model_male.pt", device)

    with torch.no_grad():
        pred = model(img_tensor, male_tensor).cpu().item()
    
    prediction = STD * pred + MEAN  # Denormalize the prediction
    prediction = round(prediction, 2)  # Round to 2 decimal places
    prediction_years = prediction / 12  # Convert months to years

    return prediction, prediction_years


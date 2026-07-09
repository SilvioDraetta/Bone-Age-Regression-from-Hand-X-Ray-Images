"""Utilities for loading the trained bone age regression model."""

import torch
from src.model.cnn_torch import BoneAgeModel  

def load_bone_age_model(weights_path="model_results/torch_model_male.pt", device="cuda"):
    """
    Load the trained bone age regression model and its weights.
    """
    device = torch.device(device)

    model = BoneAgeModel().to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    return model

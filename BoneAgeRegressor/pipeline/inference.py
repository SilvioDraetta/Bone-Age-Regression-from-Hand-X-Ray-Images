"""Inference utilities for bone age prediction from hand X-ray images."""

import sys
from pathlib import Path
import json

import torch
from torchvision import transforms
from PIL import Image

from .load_birefnet import load_birefnet
from .loader import load_bone_age_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


with open("src/config/stats.json") as f:
    stats = json.load(f)

MEAN = stats["mean"]
STD = stats["std"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def extract_object(birefnet, imagepath):
    """
    Extract the foreground object from an image using a BiRefNet model.

    The function applies the required preprocessing, performs inference to
    predict a segmentation mask, and returns the original image with the
    predicted alpha channel together with the binary mask.

    Parameters
    ----------
    birefnet : torch.nn.Module
        Pretrained BiRefNet segmentation model.

    imagepath : str or pathlib.Path
        Path to the input RGB image.

    Returns
    -------
    tuple[PIL.Image.Image, PIL.Image.Image]
        A tuple containing:

        - image : PIL.Image.Image
            Original image with the predicted alpha mask applied.

        - mask : PIL.Image.Image
            Predicted segmentation mask resized to the original image size.
    """
    image_size = (1024, 1024)

    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(imagepath).convert("RGB")

    input_images = (
        transform_image(image)
        .unsqueeze(0)
        .to("cuda")
        .half()
    )

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)

    image.putalpha(mask)

    return image, mask


def run_bone_age_inference(image_path, sex_bool, device="cuda"):
    """
    Predict bone age from a hand X-ray image and subject sex.

    The function segments the hand using BiRefNet, preprocesses the image,
    loads the trained bone age regression model, performs inference, and
    converts the normalized prediction back to months and years.

    Parameters
    ----------
    image_path : str or pathlib.Path
        Path to the input hand X-ray image.

    sex_bool : bool
        Subject sex, where True represents male and False represents female.

    device : str, optional
        Device used for model inference. Default is "cuda".

    Returns
    -------
    tuple[float, float]
        Predicted bone age in months and years.
    """
    model_biref = load_birefnet()
    image, _ = extract_object(model_biref, image_path)

    image = image.convert("L")

    img_tensor = transform(image).unsqueeze(0).to(device)

    male_tensor = torch.tensor(
        [[1.0 if sex_bool else 0.0]],
        device=device
    )

    model = load_bone_age_model(
        "notebook/model_results/torch_model_male.pt",
        device
    )

    with torch.no_grad():
        pred = model(img_tensor, male_tensor).cpu().item()

    prediction = STD * pred + MEAN
    prediction = round(prediction, 2)

    prediction_years = prediction / 12

    return prediction, prediction_years


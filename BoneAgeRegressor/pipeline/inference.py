import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

import torch
from .load_birefnet import load_birefnet
import json
from .loader import load_bone_age_model
from torchvision import transforms
from PIL import Image

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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath).convert("RGB")

    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask

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


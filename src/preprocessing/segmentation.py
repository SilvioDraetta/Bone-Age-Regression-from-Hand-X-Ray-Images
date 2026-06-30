import os
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms

def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath).convert("RGB")

    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask


def segmentation_folder(birefnet, input_dir, output_dir):
    """
    Process all images in a directory using a BiRefNet segmentation model and
    save the extracted objects to an output folder.

    This function iterates through every file in `input_dir`, applies the
    `extract_object` function using the provided `birefnet` model, and saves
    the resulting segmented image to `output_dir`. Output filenames preserve
    the original name with the suffix "_seg" and use PNG format.

    Parameters
    ----------
    birefnet : object
        The BiRefNet model instance used for object extraction.
    input_dir : str
        Path to the directory containing input images.
    output_dir : str
        Path to the directory where segmented images will be saved. The
        directory is created if it does not already exist.

    Returns
    -------
    None
        The function prints a completion message when all images have been processed.
    """
    os.makedirs(output_dir, exist_ok=True)

    ext = ".png"
    seg = "_seg"

    for filename in tqdm(os.listdir(input_dir)):
        
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, os.path.splitext(filename)[0] + seg + ext)

        result = extract_object(birefnet, in_path)[0]
        result.save(out_path)

    print("✔️ Completed! All images processed.")

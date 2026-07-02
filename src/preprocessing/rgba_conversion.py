import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def rgba_to_grayscale(img):
    """
    Converte RGBA → grayscale usando alpha e restituisce immagine L (1 canale)
    """

    rgba = np.array(img).astype(np.float32)

    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4] / 255.0

    # applica alpha
    rgb = rgb * alpha

    # luminanza (standard percettivo)
    gray = (
        0.299 * rgb[..., 0] +
        0.587 * rgb[..., 1] +
        0.114 * rgb[..., 2]
    )

    gray = gray.astype(np.uint8)

    return Image.fromarray(gray, mode="L")


def convert_folder(input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

    for f in tqdm(files):

        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)

        img = Image.open(in_path).convert("RGBA")

        gray_img = rgba_to_grayscale(img)

        gray_img.save(out_path)



"""Utilities for converting segmented RGBA images to grayscale images."""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def rgba_to_grayscale(img):
    """
    Convert an RGBA segmented image to a grayscale image.

    The function applies the alpha channel as a segmentation mask to the
    RGB channels, removing the transparent background. The masked RGB image
    is then converted to grayscale using the standard luminance formula.

    Parameters
    ----------
    img : PIL.Image.Image
        Input RGBA image.

    Returns
    -------
    PIL.Image.Image
        Grayscale image in ``L`` mode.
    """

    rgba = np.array(img).astype(np.float32)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4] / 255.0

    rgb = rgb * alpha

    gray = (
        0.299 * rgb[..., 0] +
        0.587 * rgb[..., 1] +
        0.114 * rgb[..., 2]
    )

    gray = gray.astype(np.uint8)

    return Image.fromarray(gray, mode="L")


def convert_folder(input_dir, output_dir):
    """
    Convert all PNG images in a folder to grayscale.

    Each image is read as RGBA, converted to grayscale using
    ``rgba_to_grayscale()``, and saved in the output directory while
    preserving the original filename.

    Parameters
    ----------
    input_dir : str
        Directory containing the input PNG images.

    output_dir : str
        Directory where the converted grayscale images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

    for f in tqdm(files):

        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f)

        img = Image.open(in_path).convert("RGBA")

        gray_img = rgba_to_grayscale(img)

        gray_img.save(out_path)



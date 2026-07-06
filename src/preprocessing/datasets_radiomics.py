import cv2
import numpy as np
import SimpleITK as sitk


def load_segmented_for_radiomics(path):
    """
    Load a segmented radiography and prepare it for PyRadiomics.

    The image is read in grayscale, resized to 224x224 pixels, and
    converted to a SimpleITK image. A binary mask is generated from the
    non-zero pixels and resized accordingly.

    Parameters
    ----------
    path : str
        Path to the segmented radiography.

    Returns
    -------
    tuple[SimpleITK.Image, SimpleITK.Image]
        The resized grayscale image and its corresponding binary mask.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (img > 0).astype(np.uint8)

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    image_sitk = sitk.GetImageFromArray(img.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))

    return image_sitk, mask_sitk


from tqdm import tqdm
import pandas as pd

def extract_radiomics_features(df, extractor, n_features=50):
    """
    Extract the first n original PyRadiomics features for each image.
    """

    all_features = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting radiomics features"
    ):

        image, mask = load_segmented_for_radiomics(row["path"])

        result = extractor.execute(image, mask, label=1)

        features = {
            k: float(v)
            for k, v in result.items()
            if k.startswith("original_")
        }

        selected_features = dict(list(features.items())[:n_features])

        selected_features["id"] = row["id"]
        selected_features["boneage"] = row["boneage"]
        selected_features["male"] = row["male"]

        all_features.append(selected_features)

    return pd.DataFrame(all_features)
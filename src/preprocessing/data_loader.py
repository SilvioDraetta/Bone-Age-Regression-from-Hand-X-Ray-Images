import tensorflow as tf
import cv2
import numpy as np
import SimpleITK as sitk

def load_image(path, label):
    """
    Load and preprocess an image for model training.

    The function reads an image from the given path, decodes it as a
    grayscale image, resizes it to 224x224 pixels, and normalizes the
    pixel values to the range [0, 1].
    ----------
    Parameters
    ----------
    path : tf.Tensor
        Path to the image file.

    label : tf.Tensor
        Corresponding label associated with the image (e.g. bone age).

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        A tuple containing:
        
        - img : tf.Tensor
            Preprocessed image with shape (224, 224, 1) and normalized
            pixel values.
        
        - label : tf.Tensor
            Original image label.
    """

    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)   
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    return img, label

#DA IMPLEMENTARE SPLIT DEL DATASET
def create_dataset_tf(dataset, load_image, num_samples=None, seed=42):
    """
    Create an optimized TensorFlow Dataset pipeline for model training.

    The function optionally selects a random subset of samples from the
    input dataset, applies image loading and preprocessing, shuffles the
    data, groups samples into batches, and enables asynchronous data
    loading through prefetching.

    Parameters
    ----------
    dataset : tf.data.Dataset
        TensorFlow dataset containing image paths and corresponding labels.

    load_image : callable
        Function used to load and preprocess images. It must accept a
        dataset element and return a tuple:
        (image_tensor, label).

    num_samples : int, optional
        Number of samples to keep from the dataset. If None, the entire
        dataset is used.

    seed : int, default=42
        Random seed used for dataset shuffling to ensure reproducibility.

    Returns
    -------
    tf.data.Dataset
        Prepared TensorFlow dataset ready to be used for model training.

    """
    dataset_length = dataset.cardinality().numpy()
    if num_samples is not None:
        num_samples = min(num_samples, dataset_length)
        dataset = dataset.shuffle(dataset_length, seed=seed)
        dataset = dataset.take(num_samples)
        

    dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset


def load_image_segmented(path, label):
    """
    Load and preprocess a segmented RGBA image for model inference.

    The function reads an RGBA PNG image from the given path, decodes it
    with four channels (R, G, B, A), and resizes it to 224×224 pixels.
    The alpha channel contains the segmentation mask, which is applied to
    the RGB channels to isolate the segmented region. The masked RGB image
    is then converted to grayscale and normalized to the range [0, 1].

    Parameters
    ----------
    path : tf.Tensor
        Path to the RGBA image file.

    label : tf.Tensor
        Corresponding label associated with the image (e.g., bone age).

    Returns
    -------
    tuple[tf.Tensor, tf.Tensor]
        A tuple containing:

        - img_segmented : tf.Tensor
            Grayscale segmented image with shape (224, 224, 1), normalized
            to the range [0, 1].

        - label : tf.Tensor
            Original image label.
    """

    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=4)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0

    rgb = img[:, :, :3]
    alpha = img[:, :, 3:]

    img_segmented = rgb * alpha
    img_segmented = tf.image.rgb_to_grayscale(img_segmented)

    return img_segmented, label

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

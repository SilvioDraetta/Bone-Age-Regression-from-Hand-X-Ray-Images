"""Tests for the Neural Network dataset preprocessing module."""
import sys
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.datasets import (
    load_image,
    create_dataset_tf,
    load_image_segmented,
    BoneAgeDataset
)

class TestDatsets(unittest.TestCase):
    """Tests for Datasets module """
    def setUp(self):
        "Create test of what we need"
        self.path = Path("tests/test_data/1377.png")
        self.path_seg = Path("tests/test_data/1377_seg.png")
        self.label = 120.
        self.img, self.label = load_image(str(self.path), self.label)
        self.img_seg, self.label = load_image_segmented(str(self.path_seg), self.label)

    def test_shape(self):
        """Test that image and mask have size 224x224"""
        self.assertEqual(self.img.shape, (224, 224, 1))

    def test_shape_seg(self):
        """Test that image and mask have size 224x224 for the seg path"""
        self.assertEqual(self.img_seg.shape, (224, 224, 1))

    def test_typeofimage(self):
        """Test that image and mask is a tensor of tf"""
        self.assertIsInstance(self.img, tf.Tensor)

    def test_typeofimage_seg(self):
        """Test that image and mask is a tensor of tf for the seg path"""
        self.assertIsInstance(self.img_seg, tf.Tensor)

    def test_normalize(self):
        """Test if the image is normalised"""
        self.assertTrue(np.all((self.img >= 0) & (self.img <= 1)))

    def test_normalize_seg(self):
        """Test if the image is normalised for the seg path"""
        self.assertTrue(np.all((self.img_seg >= 0) & (self.img_seg <= 1)))

    def test_label(self):
        """Test that the label is preserved."""
        self.assertEqual(float(self.label), 120.0)

    def test_output_has_one_channel(self):
        """Test that the segmented image is grayscale."""
        self.assertEqual(self.img_seg.shape[-1], 1)
    def test_image_is_not_empty(self):
        """Test that the image is not completely black."""
        self.assertGreater(float(tf.reduce_sum(self.img)), 0.0)


def dummy_load_image(path, label):
    image = tf.ones((224, 224, 1), dtype=tf.float32)
    return image, label


class TestCreateDatasetTF(unittest.TestCase):
    """Unit tests for the TensorFlow dataset creation pipeline."""

    def setUp(self):
        """Create a dummy dataset for the tests."""
        paths = tf.constant([
            "img1.png", "img2.png", "img3.png", "img4.png",
            "img5.png", "img6.png", "img7.png", "img8.png",
            "img9.png", "img10.png"
        ])

        labels = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             dtype=tf.float32)

        self.dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def test_returns_tf_dataset(self):
        """Verify that the function returns a TensorFlow Dataset object."""
        result = create_dataset_tf(
            self.dataset,
            dummy_load_image
        )

        self.assertIsInstance(result, tf.data.Dataset)

    def test_image_shape_after_map(self):
        """Verify that mapped images have the expected shape and label type."""
        result = create_dataset_tf(
            self.dataset,
            dummy_load_image
        )

        images, labels = next(iter(result))

        self.assertEqual(images.shape[1:], (224, 224, 1))
        self.assertEqual(labels.dtype, tf.float32)

    def test_batch_size(self):
        """Verify that batches contain at most 32 samples and matching labels."""
        result = create_dataset_tf(
            self.dataset,
            dummy_load_image
        )

        images, labels = next(iter(result))

        self.assertLessEqual(images.shape[0], 32)
        self.assertEqual(images.shape[0], labels.shape[0])

    def test_num_samples(self):
        """Verify that the dataset is correctly reduced to the requested number of samples."""
        result = create_dataset_tf(
            self.dataset,
            dummy_load_image,
            num_samples=5
        )

        total_samples = 0

        for images, labels in result:
            total_samples += images.shape[0]

        self.assertEqual(total_samples, 5)

    def test_num_samples_larger_than_dataset(self):
        """Verify that requesting more samples than available returns the whole dataset."""
        result = create_dataset_tf(
            self.dataset,
            dummy_load_image,
            num_samples=100
        )

        total_samples = 0

        for images, labels in result:
            total_samples += images.shape[0]

        self.assertEqual(total_samples, 10)

class TestBoneAgeDataset(unittest.TestCase):
    """Unit tests for the BoneAgeDataset PyTorch dataset."""

    def setUp(self):
        """Create a temporary image and a dummy DataFrame for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.img_path = Path(self.temp_dir.name) / "test_img.png"

        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(self.img_path)

        self.df_seg = pd.DataFrame({
            "path": [str(self.img_path)],
            "male": [1],
            "boneage": [120],
            "id": [1377]
        })

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.dataset = BoneAgeDataset(
            self.df_seg,
            transform=self.transform
        )

    def tearDown(self):
        """Remove temporary files created during the test."""
        self.temp_dir.cleanup()

    def test_len(self):
        """Verify that the dataset length matches the DataFrame length."""
        self.assertEqual(len(self.dataset), 1)

    def test_getitem_returns_dict(self):
        """Verify that __getitem__ returns a dictionary."""
        sample = self.dataset[0]

        self.assertIsInstance(sample, dict)

    def test_sample_keys(self):
        """Verify that the returned sample contains all expected keys."""
        sample = self.dataset[0]

        self.assertIn("image", sample)
        self.assertIn("male", sample)
        self.assertIn("boneage", sample)
        self.assertIn("id", sample)

    def test_image_is_tensor_with_correct_shape(self):
        """Verify that the transformed image is a tensor with the expected shape."""
        sample = self.dataset[0]

        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertEqual(sample["image"].shape, (3, 224, 224))

    def test_male_is_float_tensor(self):
        """Verify that the sex is returned as a float tensor."""
        sample = self.dataset[0]

        self.assertIsInstance(sample["male"], torch.Tensor)
        self.assertEqual(sample["male"].dtype, torch.float32)
        self.assertEqual(sample["male"].item(), 1.0)

    def test_boneage_is_float_tensor(self):
        """Verify that the bone age is returned as a float tensor."""
        sample = self.dataset[0]

        self.assertIsInstance(sample["boneage"], torch.Tensor)
        self.assertEqual(sample["boneage"].dtype, torch.float32)
        self.assertEqual(sample["boneage"].item(), 120.0)

    def test_id_is_int(self):
        """Verify that the image identifier is returned as an integer."""
        sample = self.dataset[0]

        self.assertIsInstance(sample["id"], int)
        self.assertEqual(sample["id"], 1377)


if __name__ == "__main__":
    unittest.main()


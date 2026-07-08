"""Tests for the rgba_conversion module."""

import unittest
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import os

from src.preprocessing.rgba_conversion import(
    rgba_to_grayscale,
    convert_folder
)

class TestRGBA_Conversion(unittest.TestCase):
    """
    Unit tests for RGBA image conversion utilities.
    """
    def setUp(self):
        """
        Create a test image and temporary input/output directories.
        """
        self.path = Path("tests/test_data/1377_seg.png")
        self.img = Image.open(self.path)
        self.gray = rgba_to_grayscale(self.img)
        self.input_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()
        self.img.save(Path(self.input_dir.name) / "test.png")
        convert_folder(self.input_dir.name, self.output_dir.name)

    def tearDown(self):
        """
        Remove the temporary directories created for the tests.
        """
        self.input_dir.cleanup()
        self.output_dir.cleanup()

    def test_return_PIL_img(self):
        """
        Verify that the conversion returns a PIL Image object.
        """
        self.assertIsInstance(self.gray, Image.Image)

    def test_type(self):
        """
        Verify that the output image uses uint8 pixel values.
        """
        gray_array = np.array(self.gray)
        self.assertEqual(gray_array.dtype, np.uint8)

    def test_channel(self):
        """
        Verify that the converted image is in grayscale (L mode).
        """
        self.assertEqual(self.gray.mode, "L")

    def test_folder_length(self):
        """
        Verify that the number of converted images matches the number of
        input images.
        """
        input_files = list(Path(self.input_dir.name).glob("*.png"))
        output_files = list(Path(self.output_dir.name).glob("*.png"))

        self.assertEqual(len(output_files), len(input_files))
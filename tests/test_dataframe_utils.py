"""Tests for the dataframe_utils module."""

import unittest
from pathlib import Path
import tempfile

from PIL import Image
import pandas as pd

from src.utils.dataframe_utils import create_dataframe


class TestCreateDataframe(unittest.TestCase):
    """Tests for the create_dataframe function."""

    def setUp(self):
        """Create temporary test data before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path_temp_dir = Path(self.temp_dir.name)

        self.image_folder = self.path_temp_dir / "images"
        self.image_folder.mkdir()

        self.path = self.path_temp_dir / "test_create_dataframe.csv"

        self.input_df = pd.DataFrame({
            "id": [1, 2, 3],
            "boneage": [120, 130, 140],
            "male": [True, False, True]
        })

        self.input_df.to_csv(self.path, index=False)

        for image_id in self.input_df["id"]:
            image = Image.new("RGB", (224, 224), color="white")
            image.save(self.image_folder / f"{image_id}.png")

        self.df = create_dataframe(str(self.path), str(self.image_folder))

    def tearDown(self):
        """Remove temporary test data."""
        self.temp_dir.cleanup()

    def test_dataframe(self):
        """Test that the output is a DataFrame."""
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_columns(self):
        """Test that required columns are present."""
        self.assertIn("id", self.df.columns)
        self.assertIn("path", self.df.columns)
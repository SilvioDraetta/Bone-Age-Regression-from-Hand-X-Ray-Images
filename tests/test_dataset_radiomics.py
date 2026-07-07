"""Tests for the radiomics dataset preprocessing module."""


import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk



from src.preprocessing.datasets_radiomics import (
    extract_radiomics_features,
    load_segmented_for_radiomics,
)


class FakeExtractor:
    """Fake PyRadiomics extractor used for unit tests"""

    def execute(self, image, mask, label=1):
        """Return fake radiomics features"""
        return {
            "original_feature_1": 1.0,
            "original_feature_2": 2.0,
            "original_feature_3": 3.0,
            "diagnostics_info": 999.0,
        }


class TestRadiomics(unittest.TestCase):
    """Tests for radiomics preprocessing functions"""

    def setUp(self):
        """Create test image, mask, dataframe and fake extractor"""
        self.path = Path("tests/test_data/1377_seg.png")
        self.img, self.mask = load_segmented_for_radiomics(str(self.path))

        self.df = pd.DataFrame(
            {
                "id": [1],
                "boneage": [120],
                "path": [str(self.path)],
                "male": [True],
            }
        )

        self.extractor = FakeExtractor()
        self.df_features = extract_radiomics_features(self.df, self.extractor)

    def test_shape(self):
        """Test that image and mask have size 224x224"""
        self.assertEqual(self.img.GetSize(), (224, 224))
        self.assertEqual(self.mask.GetSize(), (224, 224))

    def test_mask_is_binary(self):
        """Test that the mask contains only 0 and 1"""
        mask_np = sitk.GetArrayFromImage(self.mask)
        values = np.unique(mask_np)

        self.assertTrue(np.all(np.isin(values, [0, 1])))

    def test_typeofimage(self):
        """Test that image and mask are SimpleITK images"""
        self.assertIsInstance(self.img, sitk.Image)
        self.assertIsInstance(self.mask, sitk.Image)

    def test_dataframe(self):
        """Test that extracted features are returned as a DataFrame"""
        self.assertIsInstance(self.df_features, pd.DataFrame)

    def test_df_columns(self):
        """Test that input DataFrame contains required columns"""
        expected_columns = {"id", "boneage", "path", "male"}

        self.assertTrue(expected_columns.issubset(self.df.columns))

    def test_original_features(self):
        """Test that extracted radiomics features start with original_"""
        metadata_columns = {"id", "male", "boneage", "path"}

        feature_columns = [
            col for col in self.df_features.columns
            if col not in metadata_columns
        ]

        self.assertTrue(
            all(col.startswith("original_") for col in feature_columns)
        )

    def test_diagnostics_removed(self):
        """Test that diagnostics features are not included"""
        self.assertNotIn("diagnostics_info", self.df_features.columns)


if __name__ == "__main__":
    unittest.main()
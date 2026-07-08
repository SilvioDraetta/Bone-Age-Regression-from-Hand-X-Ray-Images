"""Tests for the scaling module."""

# pylint: disable=no-member
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.preprocessing.scaling import (
    scaling_data,
    scaling_data_torch
)


class TestScaling(unittest.TestCase):
    """Unit tests for TensorFlow and PyTorch scaling functions."""

    def setUp(self):
        """Create small dummy DataFrames and apply TensorFlow scaling."""
        self.path = Path("tests/test_data/1377.png")
        self.path_val = Path("tests/test_data/1386.png")
        self.path_test = Path("tests/test_data/1377_seg.png")

        self.df = pd.DataFrame({
            "path": [str(self.path), str(self.path_val)],
            "boneage": [120., 180.]
        })

        self.df_val = pd.DataFrame({
            "path": [str(self.path_val)],
            "boneage": [150.]
        })

        self.df_test = pd.DataFrame({
            "path": [str(self.path_test)],
            "boneage": [160.]
        })

        self.dataset, self.dataset_val, self.scaler = scaling_data(
            self.df,
            self.df_val
        )

    def test_dataframe_tf(self):
        """Test that the returned objects are TensorFlow datasets."""
        self.assertIsInstance(self.dataset, tf.data.Dataset)
        self.assertIsInstance(self.dataset_val, tf.data.Dataset)

    def test_scaler_tf(self):
        """Test that the returned scaler is a fitted StandardScaler."""
        self.assertIsInstance(self.scaler, StandardScaler)
        self.assertTrue(hasattr(self.scaler, "mean_"))
        self.assertTrue(hasattr(self.scaler, "scale_"))

    def test_scaled_train_labels_mean_tf(self):
        """Test that training labels are standardized with mean close to zero."""
        labels = []

        for _, label in self.dataset:
            labels.append(label.numpy()[0])

        self.assertAlmostEqual(np.mean(labels), 0.0, places=6)

    def test_dataset_tf_content(self):
        """Test that TensorFlow dataset contains image paths and float labels."""
        path, label = next(iter(self.dataset))

        self.assertIsInstance(path.numpy().decode("utf-8"), str)
        self.assertEqual(label.dtype, tf.float32)

    def test_scaling_data_torch_returns_dataframes(self):
        """Test that PyTorch scaling returns DataFrames and a scaler."""
        df_scaled, df_val_scaled, df_test_scaled, scaler = scaling_data_torch(
            self.df.copy(),
            self.df_val.copy(),
            self.df_test.copy()
        )

        self.assertIsInstance(df_scaled, pd.DataFrame)
        self.assertIsInstance(df_val_scaled, pd.DataFrame)
        self.assertIsInstance(df_test_scaled, pd.DataFrame)
        self.assertIsInstance(scaler, StandardScaler)

    def test_scaling_data_torch_mean(self):
        """Test that PyTorch training labels have mean close to zero."""
        df_scaled, _, _, _ = scaling_data_torch(
            self.df.copy(),
            self.df_val.copy(),
            self.df_test.copy()
        )

        self.assertAlmostEqual(
            df_scaled["boneage"].mean(),
            0.0,
            places=6
        )

    def test_scaling_data_torch_column_exists(self):
        """Test that the boneage column is preserved after scaling."""
        df_scaled, df_val_scaled, df_test_scaled, _ = scaling_data_torch(
            self.df.copy(),
            self.df_val.copy(),
            self.df_test.copy()
        )

        self.assertIn("boneage", df_scaled.columns)
        self.assertIn("boneage", df_val_scaled.columns)
        self.assertIn("boneage", df_test_scaled.columns)
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.preprocessing.segmentation import segmentation_folder


class TestSegmentationFolder(unittest.TestCase):
    """Unit tests for segmentation_folder."""

    def setUp(self):
        """Create temporary input and output directories."""
        self.input_dir = tempfile.TemporaryDirectory()
        self.output_dir = tempfile.TemporaryDirectory()

        self.image = Image.new("RGB", (64, 64), color="white")
        self.image.save(Path(self.input_dir.name) / "test.png")

        self.fake_model = object()

    def tearDown(self):
        """Remove temporary directories."""
        self.input_dir.cleanup()
        self.output_dir.cleanup()

    @patch("src.preprocessing.segmentation.extract_object")
    def test_output_file_created(self, mock_extract):
        """Test that a segmented image is saved."""
        segmented = Image.new("RGBA", (64, 64))
        mask = Image.new("L", (64, 64))

        mock_extract.return_value = (segmented, mask)

        segmentation_folder(
            self.fake_model,
            self.input_dir.name,
            self.output_dir.name
        )

        output = Path(self.output_dir.name) / "test_seg.png"

        self.assertTrue(output.exists())

    @patch("src.preprocessing.segmentation.extract_object")
    def test_number_of_output_files(self, mock_extract):
        """Test that one output image is produced for each input image."""
        segmented = Image.new("RGBA", (64, 64))
        mask = Image.new("L", (64, 64))

        mock_extract.return_value = (segmented, mask)

        segmentation_folder(
            self.fake_model,
            self.input_dir.name,
            self.output_dir.name
        )

        input_files = list(Path(self.input_dir.name).glob("*.png"))
        output_files = list(Path(self.output_dir.name).glob("*.png"))

        self.assertEqual(len(input_files), len(output_files))

    @patch("src.preprocessing.segmentation.extract_object")
    def test_extract_object_called(self, mock_extract):
        """Test that extract_object is called once for each image."""
        segmented = Image.new("RGBA", (64, 64))
        mask = Image.new("L", (64, 64))

        mock_extract.return_value = (segmented, mask)

        segmentation_folder(
            self.fake_model,
            self.input_dir.name,
            self.output_dir.name
        )

        mock_extract.assert_called_once()
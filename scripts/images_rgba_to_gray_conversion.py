import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.rgba_conversion import convert_folder

input_folder_training = "data/boneage-training-segmented"
output_folder_training = "data/boneage-training-segmented-gray"

input_folder_validation = "data/boneage-validation-segmented"
output_folder_validation = "data/boneage-validation-segmented-gray"

input_folder_test = "data/boneage-test-segmented"
output_folder_test = "data/boneage-test-segmented-gray"

convert_folder(input_folder_training, output_folder_training)
convert_folder(input_folder_validation, output_folder_validation)
convert_folder(input_folder_test, output_folder_test)
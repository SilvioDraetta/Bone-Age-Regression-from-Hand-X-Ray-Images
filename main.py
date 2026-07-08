import argparse
from src.pipeline.inference import run_segmentation
from src.pipeline.inference import run_bone_age_inference

import warnings
warnings.filterwarnings("ignore")

def parse_sex(sex_str: str) -> bool:
    """
    Convert the sex string into a boolean value.
    True  = male
    False = female

    This keeps the model interface clean and ensures
    the main script handles user input normalization.
    """
    sex_str = sex_str.lower().strip()

    if sex_str in ["male", "m", "M"]:
        return True
    if sex_str in ["female", "f", "F"]:
        return False

    raise ValueError(
        f"Invalid sex value '{sex_str}'. Use male/female, M/F or m/f."
    )


def main():
    """
    Main entry point for the Bone Age Prediction pipeline.

    - Parses command-line arguments
    - Normalizes user input
    - Calls the inference pipeline
    - Prints the final prediction
    """
    parser = argparse.ArgumentParser(
        description="Bone Age Prediction Model"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--sex",
        type=str,
        required=True,
        help="Subject sex (male/female)"
    )

    args = parser.parse_args()

    # Convert sex string to boolean
    sex_bool = parse_sex(args.sex)

    # Run inference
    prediction, prediction_years = run_bone_age_inference(args.image, sex_bool)

    print(f"Bone age prediction (months): {prediction}")
    print(f"Bone age prediction (years): {prediction_years}")



if __name__ == "__main__":
    main()

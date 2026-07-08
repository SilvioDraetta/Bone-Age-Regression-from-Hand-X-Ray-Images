# Bone Age Regression from Hand X-Ray Images

This repository contains the project developed for the **Computing Methods for Experimental Physics** course at the **University of Pisa**.

The goal of the project is to estimate the **bone age of pediatric patients** from hand radiographs using deep learning techniques. We reproduced and extended the methodology proposed in the **RSNA Pediatric Bone Age Challenge**, exploring different neural network architectures and preprocessing pipelines.

This README provides installation instructions, usage examples, an overview of the implemented pipelines, and a summary of the obtained results.

---

## Dataset

The models were trained using the dataset released for the **RSNA Pediatric Bone Age Challenge**.

**Paper**

Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al.

> **The RSNA Pediatric Bone Age Machine Learning Challenge**  
> Radiology, 2018.

https://pubs.rsna.org/doi/10.1148/radiol.2018180736

**Dataset**

https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017

---

## Features

- Bone age estimation from pediatric hand X-ray images
- Automatic hand segmentation using **BiRefNet**
- CNN-based deep learning models implemented in TensorFlow and PyTorch
- FiLM conditioning using patient sex
- Radiomics feature extraction with **PyRadiomics**
- Random Forest regression baseline
- Complete training notebooks
- Ready-to-use inference pipeline

---

## Hand Segmentation

Before training, all radiographs were segmented using **BiRefNet**, a state-of-the-art image segmentation model.

**Paper**

Zheng, P., Gao, D., Fan, D.-P., Liu, L., Laaksonen, J., Ouyang, W., & Sebe, N.

> **Bilateral Reference for High-Resolution Dichotomous Image Segmentation**  
> CAAI Artificial Intelligence Research, 2024.

https://arxiv.org/abs/2401.03407

**Repository**

https://github.com/ZhengPeng7/BiRefNet

The repository also provides utility scripts to:

- segment new datasets
- convert segmented RGBA images into grayscale images suitable for training

---

## Installation

Clone the repository:

```bash
git clone https://github.com/SilvioDraetta/Bone-Age-Regression-from-Hand-X-Ray-Images.git

cd Bone-Age-Regression-from-Hand-X-Ray-Images
```

Install the package:

```bash
python -m pip install .
```

---

## Quick Start

Predict the bone age of a new hand radiograph:

```bash
python main.py \
    --image data/PathToYourImage/image.png \
    --sex female
```

For additional examples, see the `demo/` folder.

---

## Pipeline

The default inference pipeline consists of:

1. Hand segmentation using **BiRefNet**
2. RGBA → Grayscale conversion
3. Bone age prediction using the best-performing FiLM CNN model

---

## Models Evaluated

Several approaches were investigated during the project.

| Model | Description | Test MAE |
|:------|:------------|---------:|
| CNN (TensorFlow) | Trained on original radiographs | **13.61** |
| CNN (TensorFlow) | Tested on segmented images | **22.68** |
| CNN (PyTorch) | Trained on segmented images | **13.07** |
| CNN (PyTorch) | Tested on original images | **16.18** |
| CNN + FiLM | Segmented images + patient sex | **9.45** |
| Random Forest + Radiomics | PyRadiomics features extracted from segmented images | **23.51** |

The first TensorFlow model performed poorly on segmented images, suggesting that it relied on contextual information outside the hand, especially for younger patients.

Training exclusively on segmented images significantly improved the robustness of the CNN-based models.

The best-performing architecture combines image features with the patient's sex through a **Feature-wise Linear Modulation (FiLM)** layer, achieving a **Mean Absolute Error (MAE) of 9.45 months** on the segmented test set.

As an alternative approach, handcrafted radiomics features were extracted from segmented hand radiographs using **PyRadiomics** and used to train a **Random Forest Regressor**. The model achieved a **MAE of 23.51 months**. Feature importance analysis indicated that texture descriptors—particularly **Gray Level Non-Uniformity** and **Long Run Emphasis**—together with the patient's sex were among the most informative predictors.

![Regression results](notebook/model_results/PredVsTrue.png)

---

## Repository Structure

```text
BoneAge/
│
├── data/                      # Dataset and raw images (optional)
│
├── notebook/
│   ├── model_results/
│   │   ├── 00_cnn.ipynb
│   │   ├── 01_cnn_results.ipynb
│   │   ├── 02_cnn_torch.ipynb
│   │   ├── 03_cnn_torch_results.ipynb
│   │   ├── 04_cnn_torch_male_results.ipynb
│   │   └── 05_ML.ipynb
│   └── demo.ipynb             # Example notebook
│
├── scripts/                   # Utility scripts (training, evaluation, etc.)
│
├── src/
│   ├── pycache/           # Python cache
│   ├── config/                # Configuration files
│   ├── model/                 # ML/DL models
│   ├── pipeline/              # Pipeline orchestration modules
│   ├── preprocessing/         # Image preprocessing functions
│   ├── utils/                 # Helper utilities
│   ├── visualization/         # Plotting and visualization tools
│   ├── init.py
│   └── engine.py              # Main engine for running the pipeline
│
├── tests/                     # Unit tests
│
├── .gitignore
├── LICENSE
├── log_book.txt               # Development notes
├── main.py                    # Entry point for running the project
├── pyproject.toml             # Project configuration and dependencies
└── README.md                  # Main documentation
```

---

## API Overview

The source code is organized into reusable modules.

### `src/dataset`

Contains dataset classes and preprocessing utilities used during training and inference.

### `src/models`

Implements the neural network architectures developed during the project, including the FiLM-based CNN model.

### `src/training`

Contains the training loop, loss functions, metrics, callbacks, and evaluation utilities.

### `src/utils`

General-purpose helper functions for image processing, visualization, file management, and miscellaneous utilities.

---

## Utility Scripts

The `scripts/` directory contains standalone utilities used during dataset preparation.

| Script | Description |
|:-------|:------------|
| `image_segmentation.py` | Segments hand radiographs using BiRefNet. |
| `rgba_to_grayscale.py` | Converts segmented RGBA images into grayscale images suitable for training. |

---

## Notebooks

The notebooks document the different stages of the project.

| Notebook | Description |
|:---------|:------------|
| `01_...` | Baseline TensorFlow CNN trained on original images |
| `02_...` | Data exploration and preprocessing |
| `03_...` | PyTorch CNN trained on segmented images |
| `04_...` | CNN with FiLM conditioning using patient sex |
| `05_...` | Radiomics feature extraction and Random Forest regression |

---

## Future Improvements

Possible future developments include:

- Vision Transformer (ViT) architectures
- Ensemble learning approaches
- Hyperparameter optimization
- Explainability methods (Grad-CAM, SHAP)
- Model deployment through a web interface

---

## References

**[1]**

Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al.

**The RSNA Pediatric Bone Age Machine Learning Challenge.**

*Radiology*, 2018.

https://pubs.rsna.org/doi/10.1148/radiol.2018180736

---

**[2]**

Zheng, P., Gao, D., Fan, D.-P., Liu, L., Laaksonen, J., Ouyang, W., & Sebe, N.

**Bilateral Reference for High-Resolution Dichotomous Image Segmentation.**

*CAAI Artificial Intelligence Research*, 2024.

https://arxiv.org/abs/2401.03407

https://github.com/ZhengPeng7/BiRefNet


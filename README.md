# Bone-Age-Regression-from-Hand-X-Ray-Images
This is a repository made for the exam "Computing methods for experimental physics" of University of Pisa. We attempted a pediatric bone age challenge from the subsquent article: [1] Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al. [The RSNA Pediatric Bone Age Machine Learning Challenge](https://pubs.rsna.org/doi/10.1148/radiol.2018180736). Radiology 2018; 290(2):498-503. 

Dataset used for this purpose was found on the [RSNA Pediatric Bone Age Challenge website](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017).

BoneAgeRegressor is a is a modular pipeline for bone age estimation using radiographic images of the hand.
This README serves as the main documentation for the project, covering installation, usage, pipeline structure, API overview, and development notes.

This repository contains various pipelines to create and train various DL models for the challenge. 
First of all the dataset was processed using an image segmentation model called [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) from the article [2] Zheng, P., Gao, D., Fan, D.-P., Liu, L., Laaksonen, J., Ouyang, W., & Sebe, N.  
"Bilateral Reference for High-Resolution Dichotomous Image Segmentation."  
CAAI Artificial Intelligence Research, vol. 3, pp. 9150038, 2024.

the main pipeline uses the best model we succeded to train and can be used to predict your image of a child hand radiography:
How to use in a bash terminal:
git clone https://github.com/SilvioDraetta/Bone-Age-Regression-from-Hand-X-Ray-Images.git
cd Bone-Age-Regression-from-Hand-X-Ray-Images
python pip -m pip install .

Then example of use (see demo)
python main.py --image data/PathToYourImage/image.png --sex female


scripts.images_segmentation serves to create new folders for segmented images, scripts.images_rgba_to_gray_conversion serves to convert them from RGBA to grayscale. 



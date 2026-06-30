# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation
import torch

birefnet = AutoModelForImageSegmentation.from_pretrained(
    'ZhengPeng7/BiRefNet', 
    trust_remote_code=True
    )

torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()
birefnet.half()

from src.preprocessing.segmentation import segmentation_folder

segmentation_folder(birefnet, "../data/boneage-training-dataset", "../data/boneage-segmented-training")
segmentation_folder(birefnet, "../data/boneage-validation-dataset", "../data/boneage-segmented-validation")
segmentation_folder(birefnet, "../data/boneage-test-dataset", "../data/boneage-segmented-test")
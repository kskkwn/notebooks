import torch
import torchvision

import detectron2

import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def detect_mask(img_filepath,
                pretrained_model="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                checkpoint="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
    im = cv2.imread(img_filepath)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(pretrained_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    return np.sum(outputs["instances"].pred_masks.cpu().numpy(), axis=0).astype(bool)

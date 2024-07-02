#%%
import sys
sys.path.append(r"C:\Users\USER\Desktop\LAO\Image_Recognition\yoloworld_gradio\utils")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import os
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(model="efficient_sam_vits_model"):
    if model == "efficient_sam_vits_model":
        model = build_efficient_sam_vits().to(device)
    else:
        model = build_efficient_sam_vitt().to(device)
    model.eval()
    return model

def inference_with_box(
    image: np.ndarray,
    box: np.ndarray
) -> np.ndarray:
    bbox = torch.reshape(torch.tensor(box), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    img_tensor = ToTensor()(image)
    model = load()
    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        bbox.to(device),
        bbox_labels.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
                curr_predicted_iou > max_predicted_iou
                or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou

def inference_with_boxes(
    image: np.ndarray,
    xyxy: np.ndarray
) -> np.ndarray:
    masks = []
    for [x_min, y_min, x_max, y_max] in xyxy:
        box = np.array([[x_min, y_min], [x_max, y_max]])
        mask = inference_with_box(image, box)
        masks.append(mask)
    return np.array(masks)

# def show_anns(mask, ax):
#         ax.set_autoscale_on(False)
#         img = np.ones((mask.shape[0], mask.shape[1], 4))
#         img[:, :, 3] = 0
#         color_mask = np.concatenate([np.random.random(3), [0.5]])
#         img[mask] = color_mask
#         return img

def show_anns(masks):
    # 初始化輸出圖像，完全透明 (alpha = 0)
    img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
    img[:, :, 3] = 0

    # 疊加每個遮罩到圖像上
    for mask in masks:
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[mask] = color_mask

    return img


# %%

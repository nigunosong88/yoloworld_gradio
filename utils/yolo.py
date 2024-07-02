import sys
sys.path.append(r"C:\Users\USER\Desktop\LAO\Image_Recognition\yoloworld_gradio\yolo_world")

import torch
import gradio as gr
import numpy as np
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms
import os
import PIL.Image
import cv2
import logging
import supervision as sv
import datetime
import uuid
from typing import List

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)
Config_pth='c:\\Users\\USER\\Desktop\\LAO\\Image_Recognition\\yoloworld_gradio\\yolo_world\\pretrained_weights\\yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth'
Config_py='c:\\Users\\USER\\Desktop\\LAO\\Image_Recognition\\yoloworld_gradio\\yolo_world\\pretrained_weights\\yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py'
current_directory = os.getcwd()
# current_directory=os.path.dirname(current_directory)
def setup_runner():
    # sys.path.append("C:/Users/USER/Desktop/LAO/Image_Recognition/YLW_NEW/YOLO-World-master/YOLO-World-master")
    # print(current_directory)
    # Config_pth = os.path.join(current_directory, 'yolo_world/pretrained_weights', 'yolow-v8_l_clipv2_frozen_t2iv2_bn_o365_goldg_pretrain.pth')
    # Config_py = os.path.join(current_directory, 'yolo_world/pretrained_weights', 'yolo_world_l_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
    if os.path.exists(Config_py):
        # print(Config_pth)
        try:
            # load config
            cfg = Config.fromfile(Config_py)
            cfg.work_dir = "."
            cfg.load_from = Config_pth
            print("讀取成功！")
        except Exception as e:
            print(f"讀取文件失敗: {e}")
    
    else:
        print("文件不存在，檢查路徑是否正確！")
    
    runner = Runner.from_cfg(cfg)
    runner.call_hook("before_run")
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)

    runner.model.eval()
    return runner

def run_image(
        class_names,
        runner,
        input_image,
        max_num_boxes=100,
        score_thr=0.05,
        nms_thr=0.5
):
    texts = [[t.strip()] for t in class_names.split(",")] + [[" "]]
    data_info = runner.pipeline(dict(img_id=0, img_path=input_image,
                                     texts=texts))

    data_batch = dict(
        inputs=data_info["inputs"].unsqueeze(0),
        data_samples=[data_info["data_samples"]],
    )

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        runner.model.class_names = texts
        pred_instances = output.pred_instances

    # nms
    keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep_idxs]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
    
    

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
    output.pred_instances = pred_instances

    pred_instances = pred_instances.cpu().numpy()

    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )

    labels_names= class_names.split(", ")
    labels = [labels_names[index] for index in pred_instances['labels']]
    labels = np.array(labels)
    
    image = PIL.Image.open(input_image)
    svimage = np.array(image)
    svimage = bounding_box_annotator.annotate(svimage, detections)
    svimage = label_annotator.annotate(svimage, detections, labels)
    

    return svimage[:, :, ::-1],detections.xyxy


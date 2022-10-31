# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 02:51:34 2022

@author: chrischris
"""

### images data
bs = 1                  # batch size
size = [540, 960]     # input size
sv_path = './output'    # output dir
sv_mask = True          # save mask or not
sv_result = True        # save result or not
sv_show = False         # show results
input_data = './input'  # path of input data
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136))

### segmentation model
seg_model = 'pidnet'    # pidnet or yolact_edge

### object detection model
obj_model = 'yolov5'    # yolov5 or yolov7

### inference params
# pidnet
pidnet_pretrained = './models/PIDNet/weights/best_09181228.pt'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
color_map = [0, 255]

# yolact edge

# yolov5
yolov5_pretrained = './models/YOLOv5/weights/object_5.pt'
model_yaml = './models/yolov5s.yaml'
data_yaml = './models/YOLOv5/rail.yaml'
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 1000
box_thick = 3
fontScale = 1
thickness = 2

# yolov7


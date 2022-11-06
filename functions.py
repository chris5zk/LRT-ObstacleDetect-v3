# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 03:21:13 2022

@author: chrischris
"""

### usage functions for model inferecne
from imppack import *
from hyperparams import *

# data
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def pre_trans(im, model, device='cuda'):
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]
    return im

# Model
def LoadModel(seg_model, obj_model, device='cuda'):
    # segmentation
    if seg_model == 'pidnet':
        seg_model = get_pred_model(name='pidnet-s', num_classes=2)
        seg_model = load_pretrained(seg_model, pidnet_pretrained)
        seg_model.eval()
        seg_model.to(device)
        print('========= PIDNet OK =========')
    else:
        print('error')

    # object detection
    if obj_model == 'yolov5':
        obj_model = DetectMultiBackend(weights=yolov5_pretrained, device=device, data=data_yaml)
        obj_model.eval()
        print('========= YOLOv5 OK =========')
    elif obj_model == 'yolov7':
        pass

    return seg_model, obj_model

# PIDNet
def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    return model

def seg_binary(pred, mask):
    pred = F.interpolate(pred, size=size, mode='bilinear', align_corners=True)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    # generate binary image
    for i, color in enumerate(color_map):
        mask[:, :][pred == i] = color_map[i]  # binary image of rail
    return mask

# YOLOv5
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

def obj_binary(pred, mask, names, imgsz, s):
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # generate binary image
    bxl = []
    for i, det in enumerate(pred):  # per image
        if len(det):                         # if predict the boject
            # Rescales boxes from imgsz to im0
            det[:, :4] = scale_coords(imgsz, det[:, :4], size).round()
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            # Write results
            for *xyxy, conf, cls in reversed(det):
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                bxl.append([p1, p2, int(cls), round(float(conf)*100)/100])
                w, h = (p2[1] - p1[1]), (p2[0] - p1[0])
                mask[max(0, int(p1[1]-0.25*w)): min(size[0], int(p2[1]+0.25*w)), max(0, int(p1[0]-0.25*h)): min(size[1], int(p2[0]+0.25*h))] = 1
    return mask, bxl, s

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self.new_video(path)
                ret_val, img0 = self.cap.read()
            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        print('len')
        return self.nf  # number of files

# Obstacle detection
def obstacle_detect(rail_mask, obj_mask, im0s, bxloc, names):
    intersection = rail_mask * obj_mask
    sizec = size.copy()
    sizec.append(3)
     # BGR using cv2
    alerm_mask = np.ones(sizec, np.uint8)*255
    if intersection.any() == 1:
        alerm_mask[:, :, 0] = 0
        alerm_mask[:, :, 1] = 0
    else:
        alerm_mask[:, :, 0] = 0
        alerm_mask[:, :, 2] = 0
    # rail color
    rail_col = np.array([rail_mask, rail_mask, rail_mask], dtype='uint8').transpose((1, 2, 0))
    rail_col[:, :, 0] = rail_col[:, :, 0]/255 * 255
    rail_col[:, :, 1] = rail_col[:, :, 1]/255 * 0
    rail_col[:, :, 2] = rail_col[:, :, 2]/255 * 0

    #### plot ####
    output = cv2.addWeighted(im0s, 1, rail_col, 0.5, 0)
    output = cv2.addWeighted(output, 1, alerm_mask, 0.2, 0)
    for ob in bxloc:
        # bound-box
        label = names[ob[2]]
        text = f'{label}:{ob[3]}'
        color = COLORS[ob[2]]
        fontFace = cv2.FONT_HERSHEY_COMPLEX
        labelSize = cv2.getTextSize(text, fontFace, fontScale, thickness)
        x2 = ob[0][0] + labelSize[0][0]
        y2 = ob[0][1] - labelSize[0][1]
        # plot
        output = cv2.rectangle(output, ob[0], ob[1], color, box_thick)              # box
        cv2.rectangle(output, ob[0], (x2, y2), color, cv2.FILLED)                   # text background
        cv2.putText(output, text, ob[0], fontFace, fontScale, (0,0,0), thickness)   # label

    return output

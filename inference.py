
##### Model Integrate #####
"""
    1. Load Data -> images, videos
    2. Seg model
        -input: images, video_frame
        -output: rail_masks
    3. Obj model
        -input: images, video_frame
        -output: obj_masks
    4. Detect
        -input: rail_masks, obj_masks
        -output: alerm_images
"""   
############################

from imppack import *
from hyperparams import *

if __name__ == '__main__':
    
    ##### Load model #####  
    print('========= model loading... =========')
    # segmentation
    if seg_model == 'pidnet':
        device = torch.device('cuda')
        seg_model = get_pred_model(name='pidnet-s', num_classes=2)
        seg_model = load_pretrained(seg_model, pidnet_pretrained)
        seg_model.eval()
        seg_model.to(device)
        print('========= PIDNet OK =========')
    elif seg_model == 'yolact_edge':
        pass
    
    # object detection
    if obj_model == 'yolov5':
        device = torch.device('cuda')
        obj_model = DetectMultiBackend(weights=yolov5_pretrained, device=device, data=data_yaml)
        obj_model.eval()
        stride, names, pt = obj_model.stride, obj_model.names, obj_model.pt
        print('========= YOLOv5 OK =========')
    elif obj_model == 'yolov7':
        pass
    print('========= all models loading complete =========\n')


    ##### prepare input data #####
    # Load input data as an object
    imgsz = check_img_size(size, s=32)
    dataset = LoadImages('input/images/rail', img_size=imgsz, stride=stride, auto=pt)   
    # increment output dir
    save_dir = os.path.join(sv_path, 'exp')
    save_dir = increment_path(save_dir, exist_ok=False)  
    os.mkdir(save_dir)
    
    ##### Inference ######
    print('========= start inference =========')
    with torch.no_grad():
        for path, im, im0s, vid_cap, s in dataset:   
            
            #### Pre-process input data ####
            # load image
            rail_mask = np.zeros(size).astype(np.uint8)    # canvas             
            # transform
            im = torch.from_numpy(im).to(device)
            im = im.half() if obj_model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]
                
            #### Railway Predict ####
            rail_pred = seg_model(im)
            rail_pred = F.interpolate(rail_pred, size=size, mode='bilinear', align_corners=True)
            rail_pred = torch.argmax(rail_pred, dim=1).squeeze(0).cpu().numpy()
            # generate binary image
            for i, color in enumerate(color_map):
                rail_mask[:,:][rail_pred==i] = color_map[i] # binary image of rail
    
            #### Object Predict ####
            obj_pred = obj_model(im)      
            # NMS
            obj_pred = non_max_suppression(obj_pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)       
            # generate binary image
            obj_mask = np.zeros(size).astype(np.uint8)    # canvas
            
            bxl = []
            for i, det in enumerate(obj_pred):  # per image              
                p = Path(path)
                save_path = f'sv_path/{p.name}'      # ./output/image.jpg            
                if len(det):                         # if predict the boject
                    # Rescales boxes from imgsz to im0
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], size).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write reults                
                    for *xyxy, conf, cls in reversed(det):
                        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        bxl.append([p1, p2, int(cls), round(float(conf)*100)/100])
                        w, h = (p2[1] - p1[1]), (p2[0] - p1[0])
                        obj_mask[max(0, int(p1[1]-0.25*w)) : min(size[0], int(p2[1]+0.25*w)), max(0, int(p1[0]-0.25*h)) : min(size[1], int(p2[0]+0.25*h))] = 1
            
            # object results
            print(s)
            
            #### Obstacle detection ####            
            intersection = rail_mask * obj_mask
            sizec = size.copy()
            sizec.append(3)
            # BGR using cv2
            alerm_mask = np.ones(sizec, np.uint8)*255
            if intersection.any() == 1:
                alerm_mask[:,:,0] = 0
                alerm_mask[:,:,1] = 0
            else:
                alerm_mask[:,:,0] = 0 
                alerm_mask[:,:,2] = 0   
            # rail color
            rail_col = np.array([rail_mask, rail_mask, rail_mask],dtype='uint8').transpose((1, 2, 0))
            rail_col[:, :, 0] = rail_col[:, :, 0]/255 * 255
            rail_col[:, :, 1] = rail_col[:, :, 1]/255 * 0
            rail_col[:, :, 2] = rail_col[:, :, 2]/255 * 0
            
            #### plot ####
            output = cv2.addWeighted(im0s, 1, rail_col, 0.5, 0)
            output = cv2.addWeighted(output, 1, alerm_mask, 0.2, 0)
            for ob in bxl:
                # bouond-box 
                label = obj_model.names[ob[2]]
                text = f'{label}:{ob[3]}'
                color = COLORS[ob[2]]
                fontFace = cv2.FONT_HERSHEY_COMPLEX
                labelSize = cv2.getTextSize(text, fontFace, fontScale, thickness)
                x2 = ob[0][0] + labelSize[0][0]
                y2 = ob[0][1] - labelSize[0][1]
                # plot             
                output = cv2.rectangle(output, ob[0], ob[1], color, box_thick)              # box
                cv2.rectangle(output, ob[0], (x2, y2), color, cv2.FILLED)           # text background
                cv2.putText(output, text, ob[0], fontFace, fontScale, (0,0,0), thickness)  # label
            
            #### Outputs ####
            # save mask
            if sv_mask:
                # save railway mask
                rail_path = save_dir / 'rail_mask'
                sv_img = Image.fromarray(rail_mask)
                if not os.path.exists(rail_path):
                    os.mkdir(rail_path)
                sv_img.save(rail_path / path.split("\\")[-1])
            
                # save object mask
                obj_path = save_dir / 'obj_mask'
                sv_img = Image.fromarray(obj_mask*255)
                if not os.path.exists(obj_path):
                    os.mkdir(obj_path)
                sv_img.save(obj_path / path.split("\\")[-1])
            # show results
            if sv_show:
                cv2.imshow('a', output)
                cv2.waitKey(100)
                cv2.destroyAllWindows()
            # save results
            if sv_result:
                cv2.imwrite(save_dir.__str__()+'\\'+f'{p.name}', output)
    print('========= finish inference =========')
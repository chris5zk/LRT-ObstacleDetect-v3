
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
from functions import *
from imppack import *
from hyperparams import *

if __name__ == '__main__':
    
    ##### Load model #####  
    print('========= model loading... =========')
    device = torch.device('cuda')
    seg_model, obj_model = LoadModel(seg_model,
                                     obj_model,
                                     device=device)
    stride, names, pt = obj_model.stride, obj_model.names, obj_model.pt
    print('========= all models loading complete =========\n')

    ##### prepare input data #####
    print('========= prepare data =========')
    # Load input data as an object
    imgsz = check_img_size(size, s=32)
    dataset = LoadImages(input_data,
                         img_size=imgsz,
                         stride=stride,
                         auto=pt)
    vid_path, vid_writer = [None] * bs, [None] * bs

    if sv_result:
        # increment output dir
        save_dir = os.path.join(sv_path,'exp')
        save_dir = increment_path(save_dir,
                                  exist_ok=False)
        os.mkdir(save_dir)

    ##### Inference ######
    print('========= start inference =========')
    with torch.no_grad():
        #### Pre-process input data ####
        for path, im, im0s, vid_cap, s in dataset:
            #### transform ####
            im = pre_trans(im,
                           obj_model,
                           device=device)

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            #### Railway Predict ####
            canvas = np.zeros(size).astype(np.uint8)    # canvas
            pred = seg_model(im)
            rail_mask = seg_binary(pred,
                                   canvas)

            #### Object Predict ####
            canvas = np.zeros(size).astype(np.uint8)    # canvas
            pred = obj_model(im)
            obj_mask, bxloc, s = obj_binary(pred,
                                         canvas,
                                         names,
                                         imgsz,
                                         s)

            #### Obstacle detection ####
            output = obstacle_detect(rail_mask,
                                     obj_mask,
                                     im0s,
                                     bxloc,
                                     names)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            latency = elapsed_time / 1 * 1000
            torch.cuda.empty_cache()
            FPS = 1000 / latency
            # TFPS += FPS
            s += f'Current FPS: {math.floor(FPS * 1000) / 1000}'
            print(s)

            #### Outputs ####
            # show results
            if sv_show:
                cv2.imshow('a', output)
                cv2.waitKey(100)
                cv2.destroyAllWindows()

            # save results
            if sv_result:
                p = Path(path)
                save_path = f'sv_path/{p.name}'      # ./output/image.jpg
                if dataset.mode == 'image':
                    cv2.imwrite(save_dir.__str__()+'\\'+f'{p.name}', output)
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

                elif dataset.mode =='video':
                    save_path = str(save_dir / p.name)
                    if vid_path[0] != save_path:  # new video
                        vid_path[0] = save_path
                        if isinstance(vid_writer[0], cv2.VideoWriter):
                            vid_writer[0].release()  # release previous video writer
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[0].write(output)

    print('========= finish inference =========')

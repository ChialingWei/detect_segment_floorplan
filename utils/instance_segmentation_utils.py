from detectron2.utils.visualizer import Visualizer, ColorMode, VisImage
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog

import matplotlib.pyplot as plt
import cv2
import csv
import numpy as np
import math
from collections import defaultdict
import json  
import os
from PIL import Image
from regex import D
import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F
from scipy.spatial import ConvexHull
import pickle

def resnet50_inference(resnet_weight, img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'val':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ]),
    }
    model = models.resnet50(pretrained=True).to(device)
    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)).to(device)
    model.load_state_dict(torch.load(resnet_weight))

    img = Image.open(img_path)

    validation_batch = torch.stack([data_transforms['val'](img).to(device)])
    pred_logits_tensor = model(validation_batch)                
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    return pred_probs[0][0]

def PolyArea2D(pts):
    '''
    calulating area by given points from polygon
    '''
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def segementation_polygon_shape(item_mask, pred_bbox, segm_pts):
    '''
    Compare predicted mask w/ bbox to determine the shape of object
    1. get polygon points from segmentation points
    2. distinguish the shape of polygon
        0: vertical/horizontal
        1: diagonal
        2: irregular
    3. get polygon pts list: [left_top, left_bottom, right_top, right_bottom]
    '''
    hull = ConvexHull(segm_pts)
    segm_bd = segm_pts[hull.vertices]           # get the vertices from mask
    lst_to_sort_segm_bd = segm_bd.tolist()
    lst_to_sort_segm_bd.sort()
    left_top, left_bottom, right_top, right_bottom = lst_to_sort_segm_bd[0], lst_to_sort_segm_bd[1], \
        lst_to_sort_segm_bd[-2], lst_to_sort_segm_bd[-1]
    bbox_area = abs(pred_bbox[2] - pred_bbox[0]) * abs(pred_bbox[3] - pred_bbox[1])
    if abs(left_bottom[0] - left_top[0]) <= 10 and abs(right_top[1] - left_top[1]) <= 10:
        shape = 0
    elif abs(math.dist(left_top, left_bottom) - math.dist(right_top, right_bottom)) <= 10 and \
        abs(math.dist(left_top, right_top) - math.dist(left_bottom, right_bottom)) <= 10:
        shape = 1
    elif abs(bbox_area - PolyArea2D(segm_bd)) / bbox_area <= 0.3:
        shape = 0
    else:
        shape = 2
    return segm_bd, shape, [left_top, left_bottom, right_top, right_bottom]

def pred_crop_to_list(image_path, predictor):
    '''
    output prediction to a list [{'crop_index': , 'label': , 'score': , 'bbox': , 'segmentation': , 'shape': , 'polygon': }]
    each object store in a dictionary
    '''
    crop = []
    head_tail = os.path.split(image_path)
    crop_index = head_tail[1].split("_")[0]
    im = cv2.imread(image_path)
    outputs = predictor(im) 
    pred_bbox = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    pred_class = outputs['instances'].pred_classes.cpu().numpy()
    prob = outputs['instances'].scores.cpu().numpy()
    pred_masks = np.asarray(outputs['instances'].pred_masks.to("cpu"))
    for obj_idx in range(len(pred_class)):
        crop_dic = {}
        crop_dic['crop_index'] = crop_index
        crop_dic['label'] = pred_class[obj_idx]
        crop_dic['score'] = prob[obj_idx]
        crop_dic['bbox'] = pred_bbox[obj_idx]
        item_mask = pred_masks[obj_idx]
        segmentation = np.where(item_mask == True)
        segm = np.array(segmentation)
        x = segm[1, :]
        y = segm[0, :]
        segm_pts = np.array([[x[i],y[i]] for i in range(len(x))])
        segm_pts_lst = []
        for pt in segm_pts:
            segm_pts_lst.append(pt[0])
        if len(segm_pts) > 3 and segm_pts_lst != [0] * len(segm_pts_lst):
            segm_bd, shape, poly_lst = segementation_polygon_shape(item_mask, pred_bbox[obj_idx], segm_pts)
            crop_dic['segmentation'] = segm_bd
            crop_dic['shape'] = shape
            crop_dic['polygon'] = poly_lst
            crop.append(crop_dic)
    return crop

def detectron2_inference_lst_dic(cfg_save_path, model_weights, image_path):
    '''
    infer from training models to get a list which contains prediction result
    '''
    with open(cfg_save_path, 'rb') as f:
        cfg = pickle.load(f)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    pred_crop_result = pred_crop_to_list(image_path, predictor)
    return pred_crop_result

def inference_csv_prep(pred_crop_result):
    '''
    prepare the prediction result for csv file
    vert/horizontal wall --- using bbox coordinates (calculate to sheet coordination system)
    others --- using polygons (calculate to sheet coordination system)
    output: [crop_index, label, class, left top x, left top y, left bottom x, left bottom y, 
    right top x, right top y, right bottom x, right bottom y]
    '''
    inf_result = []
    for crop in pred_crop_result:
        if crop['shape'] == 0:
            ltx, lty, lbx, lby, rtx, rty, rbx, rby = crop['bbox'][0], crop['bbox'][1], \
                                                    crop['bbox'][0], crop['bbox'][3], \
                                                    crop['bbox'][2], crop['bbox'][1], \
                                                    crop['bbox'][2], crop['bbox'][3]
        else:
            ltx, lty, lbx, lby, rtx, rty, rbx, rby = crop['polygon'][0][0], crop['polygon'][0][1], \
                                                    crop['polygon'][1][0], crop['polygon'][1][1], \
                                                    crop['polygon'][2][0], crop['polygon'][2][1], \
                                                    crop['polygon'][3][0], crop['polygon'][3][1]
        abs_ltx = ltx + ((int(crop['crop_index'])-1)//3) * 800
        abs_lty = lty + ((int(crop['crop_index'])-1)%3) * 800
        abs_lbx = lbx + ((int(crop['crop_index'])-1)//3) * 800
        abs_lby = lby + ((int(crop['crop_index'])-1)%3) * 800
        abs_rtx = rtx + ((int(crop['crop_index'])-1)//3) * 800
        abs_rty = rty + ((int(crop['crop_index'])-1)%3) * 800
        abs_rbx = rbx + ((int(crop['crop_index'])-1)//3) * 800
        abs_rby = rby + ((int(crop['crop_index'])-1)%3) * 800
        inf_result.append([crop['crop_index'], crop['label'], crop['shape'], ltx, lty, lbx, lby, rtx, rty, rbx, rby, \
            abs_ltx, abs_lty, abs_lbx, abs_lby, abs_rtx, abs_rty, abs_rbx, abs_rby])
    return inf_result

class CocoTrainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def bg_img(bg_img_path, ori_img):
    '''
    CREATE BLANK IMAGE AS BACKGROUND
    variables:
        bg_img_path: the path for blank image
        ori_img: the image you want the blank image has same size as
    return: 
        h: height of original image
        w: width of original image
    '''
    ori = cv2.imread(ori_img)
    h, w, _ = ori.shape
    blank_image = np.zeros((h,w,3), np.uint8)
    blank_image[:,:] = (255,255,255)
    cv2.imwrite(bg_img_path, blank_image)
    return h, w

def show_save_image(image_path, predictor, output_path, show):
    '''
    show and save predicted crop images which contains labels, bbox, masks
    '''
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:,:,::-1], {"thing_classes":['L door','R door','column','double door','wall']}, scale=1, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    vis_img = VisImage(v.get_image())
    vis_img.save(output_path)
    if show == True:
        plt.figure(figsize=(14,10))
        plt.imshow(v.get_image())
        plt.show()

def combine_sequence_crop(h_crop, w_crop, bg_img_path, ori_img, each_crop_path, inf_img_name, output_path):
    '''
    CONBINE SMALL CROPS TO ORIGINAL IMAGES SEQUENCIALLY;
    CREATE A BLANK IMAGE FIRST AND PASTE THE SMALL CROPS ON THAT IN AN ORDER
    variables:
        h_crop, w_crop: dimension of small crop
        bg_img_path: path for blank image
        ori_img: the image you want the blank image has same size as
        each_crop_path: path for small crops folder
        inf_img_name: name for small crop images
        output_path: the final combination image output path   
    '''
    x_offset, y_offset = 0, 0
    h, w = bg_img(bg_img_path, ori_img)
    l_img = cv2.imread(bg_img_path)      # l_img: blank image
    h_rp = math.ceil(h/h_crop)           # how much crop on height side
    w_rp = math.ceil(w/w_crop)           # how much crop on width side
    temp = -1
    for w_idx in range(1, w_rp+1):
        x_offset = w_crop * (w_idx - 1)
        for h_idx in range(1, h_rp+1):
            y_offset = h_crop * (h_idx - 1)
            s_img = cv2.imread(each_crop_path + '\\' + str(h_idx + w_idx + temp) + '_' + inf_img_name + '.jpg')       # s_img: small crop
            if np.shape(s_img) == (h%h_crop, w%w_crop, 3):
                l_img[y_offset:y_offset+h%h_crop, x_offset:x_offset+w%w_crop] = s_img   # the last crop               
            else:
                l_img[y_offset:y_offset+h_crop, x_offset:x_offset+w_crop] = s_img
        temp += (h_rp - 1)
    cv2.imwrite(output_path, l_img)

def arr_for_merge(csv_file):
    '''
    create an array from predicted csv file for merging same class objects
    '''
    arr = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            arr.append([i, row['class'], row['shape'], \
            row['abs_ltx'], 2200-float(row['abs_lty']), row['abs_lbx'], 2200-float(row['abs_lby']), \
            row['abs_rtx'], 2200-float(row['abs_rty']), row['abs_rbx'], 2200-float(row['abs_rby'])])
    return arr

def merge_reg_components(csv_file, gap_threshold):
    '''
    merge components vertical and horizontal seperately
    output:
    1. arr: walls coordinates after merging
    '''
    ori_arr = arr_for_merge(csv_file)
    del_irr = []
    for obj in ori_arr:
        if obj[2] != '0':      # remain vert/horzontal walls
            del_irr.append(obj)
    for del_obj in del_irr:
        if del_obj in ori_arr:
            ori_arr.remove(del_obj)
    del_lst = []
    arr = np.copy(ori_arr).tolist()
    for obj in arr:
        for i in range(len(arr)):
            if abs(float(obj[7]) - float(obj[3])) > abs(float(obj[6]) - float(obj[4])): 
                if float(arr[i][3]) - float(obj[7]) <= gap_threshold and abs(float(arr[i][4]) - float(obj[8])) <= gap_threshold and \
                    float(arr[i][5]) - float(obj[9]) <= gap_threshold and abs(float(arr[i][6]) - float(obj[10])) <= gap_threshold and \
                    float(arr[i][3]) > float(obj[3]) and float(arr[i][5]) > float(obj[5]) and \
                    abs(float(arr[i][7]) - float(arr[i][3])) > abs(float(arr[i][6]) - float(arr[i][4])) and \
                    obj[1] != '2' and arr[i][1] != '2':
                    arr[i][3], arr[i][4], arr[i][5], arr[i][6] = obj[3], obj[4], obj[5], obj[6]
                    arr[i][1] = '4'
                    del_lst.append(obj)
            else:
                if abs(float(arr[i][5]) - float(obj[3])) <= gap_threshold and float(arr[i][6]) - float(obj[4]) <= gap_threshold and \
                    abs(float(arr[i][9]) - float(obj[7])) <= gap_threshold and float(arr[i][10]) - float(obj[8]) <= gap_threshold and \
                    float(arr[i][6]) > float(obj[6]) and float(arr[i][10]) > float(obj[10]) and \
                    abs(float(arr[i][7]) - float(arr[i][3])) <= abs(float(arr[i][6]) - float(arr[i][4])) and \
                    obj[1] != '2' and arr[i][1] != '2':
                    arr[i][5], arr[i][6], arr[i][9], arr[i][10] = obj[5], obj[6], obj[9], obj[10]
                    arr[i][1] = '4'
                    del_lst.append(obj)
    for del_obj in del_lst:
        if del_obj in arr:
            arr.remove(del_obj)
    for obj in ori_arr:
        for i in range(len(arr)):
            if abs(float(obj[7]) - float(obj[3])) > abs(float(obj[6]) - float(obj[4])): 
                if obj[1] != '4' and obj[1] != '2' and float(arr[i][3]) <= float(obj[3]) <= float(arr[i][7]) and abs(float(obj[4]) - float(arr[i][4])) <= gap_threshold and abs(float(obj[6]) - float(arr[i][6])) <= gap_threshold and \
                    float(arr[i][5]) <= float(obj[5]) <= float(arr[i][9]) and abs(float(arr[i][7]) - float(arr[i][3])) > abs(float(arr[i][6]) - float(arr[i][4])):
                    obj[0] = arr[i][0]
                    arr.append(obj)
            else:
                if obj[1] != '4' and obj[1] != '2' and float(arr[i][6]) <= float(obj[6]) <= float(arr[i][4]) and abs(float(obj[5]) - float(arr[i][5])) <= gap_threshold and abs(float(obj[9]) - float(arr[i][9])) <= gap_threshold and \
                    float(arr[i][10]) <= float(obj[10]) <= float(arr[i][8]) and abs(float(arr[i][7]) - float(arr[i][3])) <= abs(float(arr[i][6]) - float(arr[i][4])):
                    obj[0] = arr[i][0]
                    arr.append(obj)        
    return arr

def irr_components(csv_file):
    irr_arr = arr_for_merge(csv_file)
    del_irr = []
    for obj in irr_arr:
        if obj[2] == '0':      # remain diagonal walls
            del_irr.append(obj)
    for del_obj in del_irr:
        if del_obj in irr_arr:
            irr_arr.remove(del_obj)
    return irr_arr

def merge_diagonal_components(csv_file, gap_threshold=50):
    irr_ori_arr = irr_components(csv_file)
    del_lst = []
    irr_arr = np.copy(irr_ori_arr).tolist()
    for obj in irr_arr:
        for i in range(len(irr_arr)):
            if abs(float(obj[7]) - float(obj[3])) > abs(float(obj[6]) - float(obj[4])):    # horizonal obj
                if float(irr_arr[i][3]) - float(obj[7]) <= gap_threshold and abs(float(irr_arr[i][4]) - float(obj[8])) <= gap_threshold and \
                    float(irr_arr[i][5]) - float(obj[9]) <= gap_threshold and abs(float(irr_arr[i][6]) - float(obj[10])) <= gap_threshold and \
                    float(irr_arr[i][3]) > float(obj[3]) and float(irr_arr[i][5]) > float(obj[5]) and \
                    obj[1] != '2' and irr_arr[i][1] != '2':
                    irr_arr[i][3], irr_arr[i][4], irr_arr[i][5], irr_arr[i][6] = obj[3], obj[4], obj[5], obj[6]
                    irr_arr[i][1] = '4'
                    del_lst.append(obj)
            else:
                if abs(float(irr_arr[i][5]) - float(obj[3])) <= gap_threshold and float(irr_arr[i][6]) - float(obj[4]) <= gap_threshold and \
                    abs(float(irr_arr[i][9]) - float(obj[7])) <= gap_threshold and float(irr_arr[i][10]) - float(obj[8]) <= gap_threshold and \
                    float(irr_arr[i][6]) > float(obj[6]) and float(irr_arr[i][10]) > float(obj[10]) and \
                    obj[1] != '2' and irr_arr[i][1] != '2':
                    irr_arr[i][5], irr_arr[i][6], irr_arr[i][9], irr_arr[i][10] = obj[5], obj[6], obj[9], obj[10]
                    irr_arr[i][1] = '4'
                    del_lst.append(obj)
    for del_obj in del_lst:
        if del_obj in irr_arr:
            irr_arr.remove(del_obj)
    for obj in irr_ori_arr:
        for i in range(len(irr_arr)):
            if abs(float(obj[7]) - float(obj[3])) > abs(float(obj[6]) - float(obj[4])): 
                if obj[1] != '4' and obj[1] != '2' and float(irr_arr[i][3]) <= float(obj[3]) <= float(irr_arr[i][7]) and abs(float(obj[4]) - float(irr_arr[i][4])) <= gap_threshold and abs(float(obj[6]) - float(irr_arr[i][6])) <= gap_threshold and \
                    float(irr_arr[i][5]) <= float(obj[5]) <= float(irr_arr[i][9]) and abs(float(irr_arr[i][7]) - float(irr_arr[i][3])) > abs(float(irr_arr[i][6]) - float(irr_arr[i][4])):
                    obj[0] = irr_arr[i][0]
                    irr_arr.append(obj)
            else:
                if obj[1] != '4' and obj[1] != '2' and float(irr_arr[i][6]) <= float(obj[6]) <= float(irr_arr[i][4]) and abs(float(obj[5]) - float(irr_arr[i][5])) <= gap_threshold and abs(float(obj[9]) - float(irr_arr[i][9])) <= gap_threshold and \
                    float(irr_arr[i][10]) <= float(obj[10]) <= float(irr_arr[i][8]) and abs(float(irr_arr[i][7]) - float(irr_arr[i][3])) <= abs(float(irr_arr[i][6]) - float(irr_arr[i][4])):
                    obj[0] = irr_arr[i][0]
                    irr_arr.append(obj)        
    return irr_arr

def final_dic(reg_arr, diagonal_arr):
    '''
    converting the 4 corner points to two mid points which are start point and end points for model creation
    '''
    for arr in diagonal_arr:
        reg_arr.append(arr)
    dic = defaultdict(list)
    for obj in reg_arr:
        dic[obj[0]].append(obj[1:])
    mid_pt_dic = {}
    for key, values in dic.items():
        mid_pt_dic[key] = []
        for value in values:
            value = [(float(v)/96)*9 if i > 1 else v for i,v in enumerate(value)]  
            lst = []
            lst.extend([value[0],value[1]])
            if abs(value[6]-value[2]) > abs(value[5]-value[3]):
                xmin, ymin, xmax, ymax = (value[2]+value[4])/2, (value[3]+value[5])/2, (value[6]+value[8])/2, (value[7]+value[9])/2
            else:
                xmin, ymin, xmax, ymax = (value[2]+value[6])/2, (value[3]+value[7])/2, (value[4]+value[8])/2, (value[5]+value[9])/2
            lst.extend([xmin, ymin, xmax, ymax])
            mid_pt_dic[key].append(lst)
    return mid_pt_dic
    
def final_json(csv_file, output_file):
    '''
    make final json file for revit api
    '''
    reg_arr = merge_reg_components(csv_file, 10)
    diagonal_arr = merge_diagonal_components(csv_file, gap_threshold=50)
    mid_pt_dic = final_dic(reg_arr, diagonal_arr)
    with open(output_file, "w") as outfile:
        json.dump(mid_pt_dic, outfile, indent = 4)




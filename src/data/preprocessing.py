from PIL import Image
import random
import numpy as np
import json
import cv2
import math
import os
import shutil

def random_crop(img, width, height):
    x = random.randint(0, np.shape(img)[1] - width)
    y = random.randint(0, np.shape(img)[0] - height)
    img = img[y:y+height, x:x+width]
    return img, x, y

def crop_image(save_path, name, crop_width, crop_height):
  img = Image.open(save_path + '/' + name + '.jpg')
  img = np.array(img)
  (img, x, y) = random_crop(img, crop_width, crop_height)
  img = Image.fromarray(img)
  return img, x, y

def check_p_loc(obj, x, y, crop_width, crop_height):
  num_out_points = 0
  out_idx = []
  for idx, vert in enumerate(obj['points']):
    if 0 <= vert[0]-x <= crop_width and 0 <= vert[1]-y <= crop_height:
      pass
    else: 
      num_out_points += 1
      out_idx.append(idx)
  return num_out_points, out_idx

def bd_ints(bd_lines, obj, base_idx, adj_idx, len_pts):
    intersection = []
    cross_bd_line = []
    x_y = []
    if adj_idx >= len_pts:
        adj_idx -= len_pts
    if adj_idx < 0:
        adj_idx += len_pts
    poly_x = [obj['points'][base_idx][0], obj['points'][adj_idx][0]]
    poly_y = [obj['points'][base_idx][1], obj['points'][adj_idx][1]]
    coef_p = np.polyfit(poly_x, poly_y, 1)
    poly_x.sort()
    poly_y.sort()
    if poly_x[0] == poly_x[1]:
        # object cross whole crop from up to below (no points inside crop)
        if bd_lines[0] <= poly_x[0] <= bd_lines[1] and poly_y[0] <= bd_lines[2] and poly_y[1] >= bd_lines[3]:
            intersection.append([poly_x[0], bd_lines[2]])
            intersection.append([poly_x[0], bd_lines[3]])
            cross_bd_line.append(bd_lines[2])
            cross_bd_line.append(bd_lines[3])
            x_y.append(1)
            x_y.append(1)
        # object cross crop from up bound (below points inside crop)
        elif bd_lines[0] <= poly_x[0] <= bd_lines[1] and poly_y[0] <= bd_lines[2] and bd_lines[2] <= poly_y[1] <= bd_lines[3]:
            intersection.append([poly_x[0], bd_lines[2]])
            cross_bd_line.append(bd_lines[2])
            x_y.append(1)
        # object cross crop from low bound (upper points inside crop)
        elif bd_lines[0] <= poly_x[0] <= bd_lines[1] and poly_y[1] >= bd_lines[3] and bd_lines[2] <= poly_y[0] <= bd_lines[3]:
            intersection.append([poly_x[0], bd_lines[3]])
            cross_bd_line.append(bd_lines[3])
            x_y.append(1)
    elif poly_y[0] == poly_y[1]:
        # object cross whole crop from left to right (no points inside crop)
        if bd_lines[2] <= poly_y[0] <= bd_lines[3] and poly_x[0] <= bd_lines[0] and poly_x[1] >= bd_lines[1]:
            intersection.append([bd_lines[0], poly_y[0]])
            intersection.append([bd_lines[1], poly_y[0]])
            cross_bd_line.append(bd_lines[0])
            cross_bd_line.append(bd_lines[1])
            x_y.append(0)
            x_y.append(0)
        # object cross crop from left bound (right points inside crop)
        elif bd_lines[2] <= poly_y[0] <= bd_lines[3] and poly_x[0] <= bd_lines[0] and bd_lines[0] <= poly_x[1] <= bd_lines[1]:
            intersection.append([bd_lines[0], poly_y[0]])
            cross_bd_line.append(bd_lines[0])
            x_y.append(0)
        # object cross crop from right bound (left points inside crop)        
        elif bd_lines[2] <= poly_y[0] <= bd_lines[3] and poly_x[1] >= bd_lines[1] and bd_lines[0] <= poly_x[0] <= bd_lines[1]:
            intersection.append([bd_lines[1], poly_y[0]])
            cross_bd_line.append(bd_lines[1])
            x_y.append(0)
    else:
        m = coef_p[0]
        b = coef_p[1]
        if poly_y[0] <= m*bd_lines[0]+b <= poly_y[1] and bd_lines[0] <= poly_x[1] <= bd_lines[1] and poly_x[0] <= bd_lines[0]:
            ints = [bd_lines[0], m*bd_lines[0]+b]
            intersection.append(ints)
            cross_bd_line.append(bd_lines[0])
            x_y.append(0) #'x'
        if poly_y[0] <= m*bd_lines[1]+b <= poly_y[1] and bd_lines[0] <= poly_x[0] <= bd_lines[1] and poly_x[1] >= bd_lines[1]:
            ints = [bd_lines[1], m*bd_lines[1]+b]
            intersection.append(ints)
            cross_bd_line.append(bd_lines[1])
            x_y.append(0) #'x'
        if poly_x[0] <= (bd_lines[2]-b)/m <= poly_x[1] and bd_lines[2] <= poly_y[1] <= bd_lines[3] and poly_y[0] <= bd_lines[2]:
            ints = [(bd_lines[2]-b)/m, bd_lines[2]]
            intersection.append(ints)
            cross_bd_line.append(bd_lines[2])
            x_y.append(1) #'y'
        if poly_x[0] <= (bd_lines[3]-b)/m <= poly_x[1] and bd_lines[2] <= poly_y[0] <= bd_lines[3] and poly_y[1] >= bd_lines[3]:
            ints = [(bd_lines[3]-b)/m, bd_lines[3]]
            intersection.append(ints)
            cross_bd_line.append(bd_lines[3])
            x_y.append(1) #'y'
    cross_bd_num = len(cross_bd_line)
    return intersection, cross_bd_line, x_y, cross_bd_num

def relative_vert(obj, x, y):
    for pts in obj['points']:
        pts[0] -= x
        pts[1] -= y
    return obj

def add_pts(obj, ints, bdnum):
  if bdnum == 2:
    obj['points'].append(ints[0])
    obj['points'].append(ints[1])
  return obj

def crop_img_vert(data, crop_json, x, y, crop_width, crop_height):
    id_12_idx_lst = []
    bd_lines = [x, x+crop_width, y, y+crop_height]
    for idx, obj in enumerate(data['shapes']):
        len_pts = len(obj['points'])
        num_outpts, outpts_idx = check_p_loc(obj, x, y, crop_width, crop_height)
        if num_outpts == 0:       # all points locate inside crop
            pass
        elif num_outpts == len_pts:     # all points locate outside crop
            locate_ints = [0]*len(obj['points'])
            for pt_i in range((len(obj['points']))):
                ints, _, _, _ = bd_ints(bd_lines, obj, pt_i, pt_i+1, len_pts)
                if len(ints) == 2:        # have intersection with crop
                    if pt_i+1 == len(obj['points']):
                        locate_ints[pt_i+1-4] = ints[0]
                        locate_ints[pt_i] = ints[1]
                    else:
                        if obj['points'][pt_i][1] < obj['points'][pt_i+1][1]:
                            locate_ints[pt_i] = ints[0]
                            locate_ints[pt_i+1] = ints[1]
                        else:
                            locate_ints[pt_i] = ints[1]
                            locate_ints[pt_i+1] = ints[0]

            if locate_ints == [0]*len(obj['points']):
                id_12_idx_lst.append(idx)  
            else:
                locate_ints = [pt for pt in locate_ints if pt != 0]
                obj['points'] = locate_ints 
        else:    # some points locate inside crop
            lst = list(range(len(obj['points'])))
            in_idx = [ele for ele in lst if ele not in outpts_idx] # index inside crop, ex: [3,4, 9,10, 15]
            sub_lst = []     # for createing sublist which contain index incrementing 1
            output_lst = []  # ex: [[3,4],[9,10],[15]]
            for idx, value in enumerate(in_idx):
                sub_lst.append(value)
                if idx == len(in_idx)-1 or value+1 != in_idx[idx+1]:
                    output_lst.append(sub_lst)
                    sub_lst = []
            sub_obj_lst = []
            for row in output_lst:
                sub_obj = obj.copy()
                ints1, bd_line1, x_y1, _ = bd_ints(bd_lines, sub_obj, row[0], row[0]-1, len_pts)
                ints2, bd_line2, _, _ = bd_ints(bd_lines, sub_obj, row[-1], row[-1]+1, len_pts)
                sub_obj['points'] = [ele for i in row for j, ele in enumerate(sub_obj['points']) if i == j]
                if ints1 != []:
                    sub_obj['points'].insert(0, ints1[0])
                if ints2 != []:
                    sub_obj['points'].append(ints2[0])       
                sub_obj_lst.extend(sub_obj['points'])
            if len(output_lst) == 1 and bd_line1 != bd_line2:
                if x_y1 == [0]:
                    sub_obj_lst.append([bd_line1[0], bd_line2[0]])
                else:
                    sub_obj_lst.append([bd_line2[0], bd_line1[0]])       
            obj['points'] = sub_obj_lst  
        obj = relative_vert(obj, x, y)
    
    data['shapes'] = [obj for obj_idx, obj in enumerate(data['shapes']) if obj_idx not in id_12_idx_lst]
    ele_count = len(data['shapes'])

    with open(crop_json, 'w') as file:
        json.dump(data, file, indent=2)

    return crop_json, ele_count

def sequence_crop_image(w_crop, h_crop, drawing_name, img_path, img_save_path):
  im = cv2.imread(img_path)
  h, w, _ = im.shape
  h_rp = math.ceil(h/h_crop)
  w_rp = math.ceil(w/w_crop)
  count = 0
  img = Image.open(img_path)
  img_arr = np.array(img)
  for w_idx in range(w_rp):
    for h_idx in range(h_rp):
      small_img = img_arr[h_crop*h_idx : h_crop*h_idx+h_crop, w_crop*w_idx : w_crop*w_idx+w_crop]
      img = Image.fromarray(small_img)      
      count += 1
      img.save(img_save_path + '/' + str(count) + '_' + drawing_name +'.jpg', 'JPEG')

def overlap_sequence_crop_image(w_crop, h_crop, drawing_name, img_path, img_save_path):
  im = cv2.imread(img_path)
  h, w, _ = im.shape
  h_rp = math.ceil(h/h_crop)
  w_rp = math.ceil(w/w_crop)
  count = 0
  img = Image.open(img_path)
  img_arr = np.array(img)
  for width in range(w_rp*2-1):
    for height in range(h_rp*2-1):
      small_img = img_arr[h_crop*height//2:(h_crop*height//2)+h_crop, w_crop*width//2:(w_crop*width//2)+w_crop]
      img = Image.fromarray(small_img)      
      count += 1
      img.save(img_save_path + '/' + str(count) + '_' + drawing_name +'.jpg', 'JPEG')

original_img_path = 'dataset/test_set/img'
save_no_ele = 'dataset/test_set/door_col_crop/no_door_col/'
save_contain_ele = 'dataset/test_set/door_col_crop/door_col/'
save_json = 'dataset/test_set/door_col_crop/door_col_json/'
drawing_json_root = 'dataset/test_set/door_column_json/'
crop_width = 800
crop_height = 800 

for img in os.listdir(original_img_path):
    name, jpg = os.path.splitext(img)
    for num_crop in range(100):
        print('image',name)
        (crop_img, x, y) = crop_image(original_img_path, name, crop_width, crop_height)
        drawing_json = drawing_json_root + name + '.json' 
        crop_json = save_json + name + '_' + str(num_crop) + '.json'
        shutil.copy(drawing_json, crop_json)
        with open(crop_json, "r") as file:
            data = json.load(file)
        print(x,'x',y,'y')
        crop_json, ele_count = crop_img_vert(data, crop_json, x, y, crop_width, crop_height)
        if ele_count != 0:
            crop_img.save(save_contain_ele + name + '_' + str(num_crop) + ".jpg", 'JPEG')
        else:
            crop_img.save(save_no_ele + name + '_' + str(num_crop) + ".jpg", 'JPEG')
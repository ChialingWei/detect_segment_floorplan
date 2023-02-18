from detectron2.engine import DefaultPredictor

import pickle
import csv
from utils.instance_segmentation_utils import*
from utils.object_detection_utils import*
import os
from src.data.preprocessing import sequence_crop_image

# Step 1: sequence cropping
def sequence_cropping(w_crop, h_crop, inf_img_path, save_img_path):
    for inf_img in os.listdir(inf_img_path):
        inf_img_name, _ =  os.path.splitext(inf_img)
        os.mkdir(save_img_path + inf_img_name)
        save_dir = save_img_path + inf_img_name
        img_ori_path = os.path.join(inf_img_path, inf_img)
        sequence_crop_image(w_crop, h_crop, inf_img_name, img_ori_path, save_dir)

# Step 2: detectron2 detection/segmentation   3 places to be changed
def classification_segmentation(image_path, cfg_save_path, model_path, test_threshold):
    cfg_save_path_OD = cfg_save_path
    cfg_save_path_IS = cfg_save_path

    with open(cfg_save_path_IS, 'rb') as f:
        cfg = pickle.load(f)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = test_threshold

    predictor = DefaultPredictor(cfg)
    img_name = os.path.split(image_path)[0].split("/")[-1]
    output_csv_path = 'inference/pred_csv_json/' + img_name
    final_img_path = 'inference/pred_crop/' + img_name
    os.makedirs(final_img_path)

    with open(output_csv_path + '.csv', 'w', newline='') as inf:
        header = ['crop_idx', 'class', 'shape', 'left_top_x', 'left_top_y', 'left_bottom_x', 'left_bottom_y',\
                                                'right_top_x', 'right_top_y', 'right_bottom_x', 'right_bottom_y', \
                                                'abs_ltx', 'abs_lty', 'abs_lbx', 'abs_lby', 'abs_rtx', 'abs_rty', 'abs_rbx', 'abs_rby']
        writer = csv.writer(inf)
        writer.writerow(header)
        for img_crop_name in os.listdir(image_path):
            img_crop = os.path.join(image_path, img_crop_name)
            pred_crop_result = detectron2_inference_lst_dic(cfg_save_path_IS, cfg.MODEL.WEIGHTS, img_crop)
            crop_data = inference_csv_prep(pred_crop_result)
            writer.writerows(crop_data)
            show_save_image(img_crop, predictor, final_img_path + '/' + img_crop_name, False)


# Step 3: visuailzation for prediction on drawing
def prediction_visualization(ori_img, h_crop, w_crop, each_crop_path, inf_img_name, output_path):
    bg_img_path = 'inference/blank.jpg' 
    combine_sequence_crop(h_crop, w_crop, bg_img_path, ori_img, each_crop_path, inf_img_name, output_path)

# Step 4: final output for 3D reconstruction json file
def final_output_json(image_path):
    img_name = os.path.split(image_path)[0].split("/")[-1]
    output_csv_path = 'inference/pred_csv_json/' + img_name + '.csv'
    final_json(output_csv_path, 'inference/pred_csv_json/' + img_name + '.json')

# Step 5: modify json to connect images
def mod_json(json_file, e, n):
    with open(json_file, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            for obj in value:
                obj[2] += e
                obj[4] += e
                obj[3] += n
                obj[5] += n
    with open(json_file, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)

if __name__== '__main__':
    # os.makedirs('inference/crop_image/053A_3_Bsz_3')
    # Step 1:
    # sequence_cropping(800, 800, 'inference/inf_original_img', 'inference/crop_image/')
    # Step 2:
    # cfg_save_path = "models/instance_segmentation/all_ele/all_ele_IS_cfg.pickle"
    # model_path = "models/instance_segmentation/all_ele/model_final.pth"
    # classification_segmentation(image_path="inference/crop_image/053A_3_Bsz_3/", cfg_save_path, model_path, 0.5)
    # Step 3:
    # prediction_visualization(ori_img='inference/inf_original_img/053A_3_Bsz_3.jpg', h_crop=800, w_crop=800,\
    #                         each_crop_path='inference/pred_crop/053A_3_Bsz_3', inf_img_name='053A_3_Bsz_3', \
    #                         output_path='inference/inf_img/053A_3_Bsz_3.jpg')
    # Step 4:
    final_output_json(image_path="inference/crop_image/053A_3_Bsz_3/")
    # Step 5:
    # mod_json('inference/pred_csv_json/053G_3_Bsz_6.json', -179, 6)  

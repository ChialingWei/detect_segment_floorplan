from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.config import get_cfg

import os 
import pickle

setup_logger()
# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml

# config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

output_dir = "IS_output/door_col"

device = "cuda"

train_dataset_name = "door_col_train"
train_images_path = "dataset/door_col/train/door_col"
train_json_annot_path = "dataset/door_col/door_col_train.json"

val_dataset_name = "door_col_val"
val_images_path = "dataset/door_col/val/door_col"
val_json_annot_path = "dataset/door_col/door_col_val.json"

cfg_save_path = "IS_output/door_col/door_col_IS_cfg.pickle"

num_classes = 4

register_coco_instances(name = train_dataset_name, metadata={},
json_file=train_json_annot_path, image_root=train_images_path)

register_coco_instances(name = val_dataset_name, metadata={},
json_file=val_json_annot_path, image_root=val_images_path)

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 20000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    cfg.TEST.EVAL_PERIOD = 500
    return cfg

def main():  
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device, output_dir)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    evaluator = COCOEvaluator(val_dataset_name, cfg, False, output_dir)
    val_loader = build_detection_test_loader(cfg, val_dataset_name)
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__== '__main__':
    # plot_samples(dataset_name=train_dataset_name, n=10)
    main()
    # metric_file = 'detectron2_output/metrics.json'
    # plot_acc(metric_file)












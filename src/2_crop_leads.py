# Install libraries
!pip install torch torchvision
!pip install opencv-python pandas
!pip install 'git+https://github.com/facebookresearch/detectron2.git'

import os
import cv2
import random
import pandas as pd

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader


IMAGE_DIR = "/kaggle/input/datasets/kishorejk575/ecg-lead-detection/images"
LABEL_DIR = "/kaggle/input/datasets/kishorejk575/ecg-lead-detection/labels"


classes = [
"I","II","III",
"aVR","aVL","aVF",
"V1","V2","V3","V4","V5","V6",
"rhythm"
]

class_to_id = {c:i for i,c in enumerate(classes)}

# -------------------------
# DATASET SPLIT
# -------------------------

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
random.seed(42)
random.shuffle(images)

n = len(images)

train_files = images[:int(0.8*n)]
val_files   = images[int(0.8*n):int(0.9*n)]
test_files  = images[int(0.9*n):]


def build_dataset(file_list):

    dataset_dicts = []

    for img_file in file_list:

        image_path = os.path.join(IMAGE_DIR,img_file)
        csv_path = os.path.join(LABEL_DIR,img_file.replace(".png",".csv"))

        img = cv2.imread(image_path)
        h,w = img.shape[:2]

        record = {}
        record["file_name"] = image_path
        record["image_id"] = img_file
        record["height"] = h
        record["width"] = w

        objs = []

        df = pd.read_csv(csv_path)

        for _,row in df.iterrows():

            bbox = [
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"])
            ]

            objs.append({
                "bbox":bbox,
                "bbox_mode":BoxMode.XYXY_ABS,
                "category_id":class_to_id[row["class"]]
            })

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


DatasetCatalog.register("ecg_train", lambda: build_dataset(train_files))
DatasetCatalog.register("ecg_val", lambda: build_dataset(val_files))
DatasetCatalog.register("ecg_test", lambda: build_dataset(test_files))

MetadataCatalog.get("ecg_train").set(thing_classes=classes)
MetadataCatalog.get("ecg_val").set(thing_classes=classes)
MetadataCatalog.get("ecg_test").set(thing_classes=classes)


# -------------------------
# CONFIGURATION
# -------------------------

cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.DATASETS.TRAIN = ("ecg_train",)
cfg.DATASETS.TEST = ("ecg_val",)

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
)

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000

cfg.TEST.EVAL_PERIOD = 500

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13

cfg.OUTPUT_DIR = "/kaggle/working/output_ecg"
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)


# -------------------------
# TRAINER WITH EVALUATION
# -------------------------

class ECGTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls,cfg,dataset_name,output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")

        return COCOEvaluator(dataset_name,cfg,False,output_folder)


trainer = ECGTrainer(cfg)

trainer.resume_or_load(resume=False)

trainer.train()

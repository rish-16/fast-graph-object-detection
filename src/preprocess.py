import os, json
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt
import fiftyone as fo

# dataset = fo.zoo.load_zoo_dataset(
#     "coco-2017",
#     split="validation",
#     label_types=["detections", "segmentations"],
#     classes=["person", "car"],
#     max_samples=50,
# )

IMAGES_DIR = "../dataset/data/"
LABELS_DIR = "../dataset/labels.json"
IMG_INFO_DIR = "../dataset/image_info.json"

with open(LABELS_DIR, 'r') as f:
    data = json.load(f)
    
with open(IMG_INFO_DIR, 'r') as f:
    images = json.load(f)
    
def process_json(record):
    return {
        'file_name': record['file_name'],
        'height': record['height'],
        'width': record['width'],
        'id': record['id']
    }
    
json_images = data['images']
json_annotations = data['annotations']
json_categories = data['categories']
    
MASTER_LIST = {}

for i in range(len(images['info'])):
    record = images['info'][i]
    MASTER_LIST[record['id']] = {
        'image_info': record,
        'annotations': []
    }
    
    for i, annot in enumerate(json_annotations, 0):
        if annot['image_id'] == record['id']:
            MASTER_LIST[record['id']]['annotations'].append(annot)
            
with open('../dataset/MASTERLIST.json', 'a') as f:
    json.dump(MASTER_LIST, f)
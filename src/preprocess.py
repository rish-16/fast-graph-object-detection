import os, json
import numpy as np
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

# Visualize the dataset in the FiftyOne App
# session = fo.launch_app(dataset, port=8080)

# The directory in which the dataset's images are stored
IMAGES_DIR = "../dataset/data/"
LABELS_DIR = "../dataset/labels.json"

with open(LABELS_DIR, 'r') as f:
    data = json.load(f)

print (data['annotations'][0].keys())    
print (data['annotations'][0]['image_id'])
print (data['annotations'][0]['id'])

# fig = plt.figure()
# for i, f in enumerate(os.listdir(IMAGES_DIR), 0):
    # fig.add_subplot(5, 11, i+1)
    # plt.imshow(Image.open(IMAGES_DIR + f))
    
# plt.show()
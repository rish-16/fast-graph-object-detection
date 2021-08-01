import json, os
from pprint import pprint

MASTER_DIR = '../dataset/MASTERLIST.json'

with open(MASTER_DIR, 'r') as f:
    master_data = json.load(f)
    
'''
info
    image_id : int
        image_info : dict
            file_name : string
            height : int
            width : int
            id : int
        annotations : list
            arb_entry
                segmentation : list
                area : float
                iscrowd : int
                image_id (akin to `id`) : int
                bbox : list
                category_id : int
                id : int
'''    

for img_id, info in master_data.items():
    for annot_record in info['annotations']:
        for i, seg_record in enumerate(annot_record['segmentation'], 0):
            master_data[img_id]['annotations'][i]['segmentation'] = list(map(lambda x : int(x), seg_record))
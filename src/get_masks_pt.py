import os
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages/"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks/"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        label_template_idxs = np.zeros_like(np.array(img))

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = torch.sum(masks, 0).unsqueeze(-1)
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
ds = PennFudanDataset("../dataset/data_pt/PennFudanPed/", transforms=None)

fig = plt.figure()
for i in range(1, 10):
    if i == 10:
        break
    
    img, target = ds[i]
    boxes = target['boxes']
    masks = target['masks']
    labels = target['labels']    
    
    fig.add_subplot(3, 3, i)
    plt.imshow(masks, cmap="gray")
    
plt.show()    

# for box in boxes:
    # r0 = box[0]
    # c0 = box[1]
    # r1 = box[2]
    # c1 = box[3]
    # w = r1 - r0
    # h = c1 - c0
    # rect = patches.Rectangle([r0, c0], w, h, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
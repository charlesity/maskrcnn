import os
import numpy as np
import torch
from PIL import Image
import cv2
import glob
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DatasetPlants(torch.utils.data.Dataset):
    def __init__(self, root, phase, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.img_dir = os.path.join(root, phase)
        self.img_ids = sorted(os.listdir(self.img_dir))


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.img_dir, self.img_ids[idx], 'images', self.img_ids[idx]+".png")
        mask_path = os.path.join(self.img_dir, self.img_ids[idx], 'masks', self.img_ids[idx]+".png")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((240,240))

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        mask = mask.resize((240,240))
        # convert the PIL Image into a numpy array
        mask = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            aBox = [xmin, ymin, xmax, ymax]
            degenerate_boxe = aBox[2:] <= aBox[:2]
            if degenerate_boxe.any():
                print("Degenerate")
                continue
            boxes.append(aBox)

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
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_ids)

class DatasetKaggle(torch.utils.data.Dataset):
    def __init__(self, root, phase, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_dir = os.path.join(root, phase)
        self.img_ids = sorted(os.listdir(self.img_dir))


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.img_dir, self.img_ids[idx], 'images', self.img_ids[idx]+".png")
        mask_path = os.path.join(self.img_dir, self.img_ids[idx], 'masks')
        img = Image.open(img_path).convert("RGB")
        img = img.resize((240,240))

        masks = []
        boxes = []
        for annoImg in sorted(glob.glob(os.path.join(mask_path, "*" + '.png'))):
            aMask = Image.open(annoImg)
            aMask = aMask.resize((240,240))
            aMask = np.array(aMask)
            aMask[aMask<255] = 0
            aMask[aMask >= 255] = 1
            masks.append(aMask)
            if (aMask !=0).any():
                pos = np.where(aMask >0)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                aBox = np.array([xmin, ymin, xmax, ymax])
                degenerate_boxe = aBox[2:] <= aBox[:2]
                if degenerate_boxe.any():
                    continue
                boxes.append(aBox)
        masks = np.array(masks)
        num_objs = len(masks)


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
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_ids)




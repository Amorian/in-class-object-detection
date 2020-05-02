import datetime
import os
import time

import xml.etree.ElementTree as ET

import torch
from torch import nn
import torchvision
from torchvision import transforms as T
import torchvision.models.detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from PIL import Image, ImageFile

import numpy as np

import utils
from coco_utils import get_coco, get_coco_kp
from engine import train_one_epoch, evaluate

ImageFile.LOAD_TRUNCATED_IMAGES = True
VOC_BBOX_LABEL_NAMES = ('__background__', 'laptop', 'empty chair')

class CustomDataset(object):
    def __init__(self, data_dir, transforms):
        id_list_file = os.path.join(
            data_dir, 'trainval.txt')

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.
        Args:
            i (int): The index of the example.
        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        for obj in anno.findall('object'):
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('xmin', 'ymin', 'xmax', 'ymax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int64)
        area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(bbox),), dtype=torch.int64)

        targets = {}
        targets['boxes'] = torch.from_numpy(bbox)
        targets['labels'] = torch.from_numpy(label)
        targets['image_id'] = torch.tensor([i])
        targets["area"] = torch.from_numpy(area)
        targets["iscrowd"] = iscrowd
        # Load a image
        img_file = os.path.join(self.data_dir, 'Images', id_ + '.jpg')
        img = Image.open(img_file).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, targets

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    num_classes = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = CustomDataset('Dataset', get_transform(train=True))
    dataset_test = CustomDataset('Dataset', get_transform(train=False))
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-2])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-2:])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), 'model' + str(epoch) + '.pth')

if __name__ == "__main__":
    main()

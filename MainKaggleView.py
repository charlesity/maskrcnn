import sys
import torch

from engine import evaluate_and_view
import utils

from Datasets import DatasetKaggle
import numpy as np

import pandas as pd






torch.manual_seed(3)
np.random.seed(3)

def main(dataset_root):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    model = utils.get_model_instance_segmentation(num_classes)
    model.to(device)
    model.load_state_dict(torch.load('./kaggle_weights/kaggle_weights.pth'))

    dataset_test = DatasetKaggle(dataset_root, 'val', utils.get_transform(train=True))
    data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,collate_fn=utils.collate_fn)
    evaluate_and_view(model, data_loader_test, device, 'kaggle', .3)

if __name__ == '__main__':
    print("Arguments passed ", sys.argv)
    root = sys.argv[1]
    main(root)
import sys
import torch

from engine import evaluate, evaluate_metric
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

    score_threshold = [s for s in np.arange(.10, .90, .10)]
    mask_threshold = [m for m in np.arange(.30, 1, .10)]
    kaggle_precisions =[]
    kaggle_ioU = []

    for s in score_threshold:
        print("processing score {}".format(s))
        p_row = []
        iou_row =[]
        for m in mask_threshold:
            print("processing threshold {}".format(m))
            dataset_test = DatasetKaggle(dataset_root, 'val', utils.get_transform(train=True))
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, batch_size=1, shuffle=False, num_workers=4,
                collate_fn=utils.collate_fn)
            result = evaluate_metric(model, data_loader_test, device, s, m)
            p_row.append(result[0])
            iou_row.append((result[1]))
        kaggle_precisions.append(p_row)
        kaggle_ioU.append(iou_row)
        print("Done for {} score ".format(s))

    data_kaggle_precisions = pd.DataFrame(kaggle_precisions)
    data_kaggle_iou = pd.DataFrame(kaggle_ioU)

    data_kaggle_precisions.to_csv("data_kaggle_precisions.csv")
    data_kaggle_iou.to_csv("data_kaggle_iou.csv")

if __name__ == '__main__':
    print("Arguments passed ", sys.argv)
    root = sys.argv[1]
    main(root)
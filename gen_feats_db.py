import os
import pickle
import sys
from collections import OrderedDict
import sklearn
from sklearn.decomposition import PCA

import torch
from torchvision import transforms, models

sys.path.append("../")
import time
import cv2
import numpy as np




normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def get_feat(model, img_cv2, use_cuda=True):

    _, features = model(input_var)

    if use_cuda:
        feature = features.data.cpu().numpy()
    else:
        feature = features.data.numpy()
    feature = sklearn.preprocessing.normalize(feature).flatten()
    return feature


if __name__ == '__main__':

    feats = []
    image_dir = '/home/hyer/workspace/business/Face/MJFR_reg_imgs'
    feats_pkl = "./data/lcnn29v2_mj.pkl"
    with open('./data/register.txt', 'rb') as f:
        lines = f.readlines()
        for line in lines:
            label, img_path = line.split('\n')[0].split()
            print label, img_path

            img = cv2.imread(img_path)
            feat = get_feat(model, img)
            feats.append((label, feat))

    f = open(feats_pkl, 'w')
    pickle.dump(feats, f, 0)
    f.close()

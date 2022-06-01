# coding: utf-8
import os
import cv2
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
from glob import glob
from torch.utils.data import DataLoader

from utils import get_query2set

ratio = [0.05, 0.2]
ratio_str = [str(int(x*100)) for x in ratio]
s_len = 11
img_size = 200

data_path = './data/test'
data = []
finger_list = glob(os.path.join(data_path,'*', '*'))
finger_list.sort()
for finger_path in finger_list:
    file_list = glob(os.path.join(finger_path, '*.png'))
    file_list.sort()
    data.append(file_list)

def to_rgb(x):
    if len(x.shape)==2:
        x = np.tile(np.expand_dims(x, -1), [1,1,3])
    return x

count = 1
while count<=2000:
    for i in tqdm(range(len(data))):
        try:
            # Positive
            if(random.uniform(0,1)<0.5):
                label = 1
                # load images (cv2->ndarray)
                image_path = random.choice(data[i])
                image = cv2.imread(image_path, 0)
                q, s = get_query2set(image, s_len, img_size, ratio)
            # Negative
            else:
                label = 0
                # load images (cv2->ndarray)
                image_path_1 = random.choice(data[i])
                image_1 = cv2.imread(image_path_1, 0)
                q, _ = get_query2set(image_1, s_len, img_size, ratio)
                s_idx_list = list(range(len(data)))
                s_idx_list.remove(i)
                s_idx = random.choice(s_idx_list)
                image_path_2 = random.choice(data[s_idx])
                image_2 = cv2.imread(image_path_2)
                _, s = get_query2set(image_2, s_len, img_size, ratio)
            
            if len(s)<s_len:
                continue
            q = to_rgb(q['x'])
            s = [to_rgb(si['x']) for si in s]
            
            output_path = './data/partial_test_{}/{:05d}-{:d}'.format('-'.join(ratio_str), count, label)
            if not os.path.isdir(output_path):
                os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(os.path.join(output_path, 'q.bmp'), q)
            for j in range(len(s)):
                cv2.imwrite(os.path.join(output_path, 's{:d}.bmp'.format(j+1)), s[j])
            count = count + 1
            if count>2000:
                break
        except:
            continue
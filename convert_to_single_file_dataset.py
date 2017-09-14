import json
import os
import numpy as np

with open('../storage/data/world_expo_dataset.json') as json_file:
    dataset_dict = json.load(json_file)

for data_type, base_names in dataset_dict.items():
    print(data_type)
    for base_name in base_names:
        print(base_name)
        images = np.load('../storage/data/' + base_name + '_images.npy')
        labels = np.load('../storage/data/' + base_name + '_labels.npy')
        roi = np.load('../storage/data/' + base_name + '_roi.npy')
        os.makedirs('../storage/data/' + base_name, exist_ok=True)
        for index, image in enumerate(images):
            np.save('../storage/data/' + base_name + '/image_{}'.format(index), image)
        for index, label in enumerate(labels):
            np.save('../storage/data/' + base_name + '/label_{}'.format(index), label)
        np.save('../storage/data/' + base_name + '/roi', roi)
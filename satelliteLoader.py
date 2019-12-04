import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms, datasets


class satelliteDataSet(data.Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.label_filenames = []

        for file in os.listdir(os.path.join(self.data_dir, 'labels')):
            if file.endswith('.json'):
                self.label_filenames.append(os.path.join(file))

    def __getitem__(self, idx):

        img_name = os.path.join(self.data_dir, 'images', self.label_filenames[idx].replace('.json', '.png'))
        #print(img_name)
        #image = io.imread(img_name)
        image = plt.imread(img_name).astype(float)
        #print('inital read of image: {} : {}'.format(img_name, image))

        label_name = os.path.join(self.data_dir, 'labels', self.label_filenames[idx])
        with open(label_name, "r") as file:
            label = json.load(file)
        features = label['features']['xy']

        buildingsDF = pd.DataFrame(columns=['Poly_X', 'Poly_Y', 'Type', 'UID'])
        damage_map = {
            'background': 0,
            'no-damage': 1,
            'minor-damage': 2,
            'major-damage': 3,
            'destroyed': 4,
        }
        types = []


        labels = np.zeros((image.shape[0],image.shape[1]))


        for i,feature in enumerate(features):
            if 'subtype' in feature['properties'].keys():
                type = damage_map[feature['properties']['subtype']]
            else:
                type = 1
            id = feature['properties']['uid']
            points = feature['wkt'].replace('POLYGON ((', '').replace('))','').split(', ')
            polygon = [[float(p) for p in pt.split(' ')] for pt in points]
            polygon = np.array(polygon,dtype='int32')
            cv2.fillPoly(labels, [polygon], color=type)


        #if self.transform:
        #    image = self.transform(image)
        #print('image after transform')
        #print(image)
        try:
            if image == 0:
                print(img_name)
        except:
            pass
        return image,labels

    def __len__(self):
        return len(self.label_filenames)



def show_buildings(image, buildings):
    """Show image with buildings"""
    damage_colors = {
        1: 'b',
        2: 'g',
        3: 'y',
        4: 'r',
    }
    plt.imshow(image)
    buildings['Type'] = buildings['Type'].map(damage_colors)
    polygons = buildings.groupby(['UID'])
    for name, poly in polygons:
        p = plt.Polygon(poly[['Poly_X', 'Poly_Y']], fill=True, color=poly['Type'][0])
        ax.add_patch(p)
    plt.pause(0.001)  # pause a bit so that plots are updated


label_colors = {
    1: (0,0,255),
    2: (0,255,0),
    3: (255,255,0),
    4: (255,0,0),
}

if __name__ == '__main__':
    sat_dataset = satelliteDataSet(data_dir='../data/poster')

    fig = plt.figure()

    for i in range(len(sat_dataset)):
        image, label = sat_dataset[i]


        print(i, image.shape, image.shape)
        print(np.max(image))
        print(np.min(image))

        #ax = plt.subplot(1, 1, i + 1)
        plt.tight_layout()
        #ax.set_title('Sample #{}'.format(i))
        #ax.axis('off')
        #show_buildings(**sample)
        #l = sample['labels']
        #l[l==1] = label_colors[1]
        #l[l==2] = label_colors[2]
        #l[l==3] = label_colors[3]
        #l[l==4] = label_colors[4]
        plt.imshow(label)
        plt.clim(0,4)

        #if i == 0:
        plt.show()
        #break


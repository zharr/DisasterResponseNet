import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

        for file in os.listdir(os.path.join(self.data_dir, 'train/labels')):
            if file.endswith('.json'):
                self.label_filenames.append(os.path.join(file))

    def __getitem__(self, idx):

        img_name = os.path.join(self.data_dir, 'train/images', self.label_filenames[idx].replace('.json', '.png'))
        #image = io.imread(img_name)
        image = plt.imread(img_name).astype(float)

        label_name = os.path.join(self.data_dir, 'train/labels', self.label_filenames[idx])
        with open(label_name, "r") as file:
            label = json.load(file)
        features = label['features']['xy']

        buildingsDF = pd.DataFrame(columns=['Poly_X', 'Poly_Y', 'Type', 'UID'])
        damage_map = {
            'no-damage': 1,
            'destroyed': 4,
        }
        types = []


        for i,feature in enumerate(features):
            type = damage_map[feature['properties']['subtype']]
            id = feature['properties']['uid']
            points = feature['wkt'].replace('POLYGON ((', '').replace('))','').split(', ')
            polygon = [[float(p) for p in pt.split(' ')] for pt in points]
            polygon = np.array(polygon)
            polyDF = pd.DataFrame(columns=['Poly_X', 'Poly_Y', 'Type', 'UID'])
            polyDF['Poly_X'] = polygon[:,0]
            polyDF['Poly_Y'] = polygon[:,1]
            polyDF['Type'] = type
            polyDF['UID'] = id
            buildingsDF = buildingsDF.append(polyDF)
            types.append(type)

        sample = {'image': image, 'buildings': buildingsDF}

        if self.transform:
            sample = self.transform(sample)

        return sample

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




if __name__ == '__main__':
    sat_dataset = satelliteDataSet(data_dir='../data/')

    fig = plt.figure()

    for i in range(len(sat_dataset)):
        sample = sat_dataset[i]


        print(i, sample['image'].shape, sample['buildings'].shape)

        ax = plt.subplot(1, 1, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_buildings(**sample)

        if i == 0:
            plt.show()
            break











# put all this in train
if False:
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = '../data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
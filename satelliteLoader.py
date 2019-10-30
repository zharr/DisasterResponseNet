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
                self.label_filenames.append(os.path.join(self.data_dir, 'train/labels', file))

    def __getitem__(self, idx):

        img_name = os.path.join(self.data_dir, 'train/images', self.label_filenames[idx].replace('.json', 'png'))
        image = io.imread(img_name)

        label_name = os.path.join(self.data_dir, 'train/labels', self.label_filenames[idx])
        with open(label_name, "r") as file:
            label = json.load(file)
        poly = label['features']['xy']


        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.label_filenames)



def show_buildings(image, buildings):
    """Show image with buildings"""
    plt.imshow(image)
    plt.scatter(buildings[:, 0], buildings[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated



if __name__ == '__main__':
    sat_dataset = satelliteDataSet(data_dir='../data/')

    fig = plt.figure()

    for i in range(len(sat_dataset)):
        sample = sat_dataset[i]

        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_buildings(**sample)

        if i == 3:
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
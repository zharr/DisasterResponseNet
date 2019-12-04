import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from unet_model import UNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

best_model_name = 'best_model.pt'
best_model = torch.load(best_model_name)

model = UNet()
model.load_state_dict(best_model['state_dict'])
model.eval()
model.to(device)

test_dir = '../data/test/images/'
test_images = [os.path.join(test_dir, x) for x in os.listdir(test_dir)]

counter = 0
i = 0
for i in range(len(test_images)):
    test_image_one = test_images[i]
    #if 'post' not in test_image_one:
    #    i += 1
    #    continue
    #counter += 1
    #i += 1
    print(i, test_image_one)
    image_one = plt.imread(test_image_one).astype(float)
    image_one = torch.Tensor(image_one).to(device)
    image_one = image_one.unsqueeze(0).permute(0,3,1,2)

    output = model(image_one)
    _, preds = torch.max(output, 1)
    preds = preds.squeeze(0)
    plt.imshow(preds.cpu())
    plt.clim(0,3)
    plt.show()
    #print(preds)
    #print(torch.max(preds))
    plt.savefig(os.path.join('test', os.path.basename(test_image_one)))

import time
import copy
import os
import numpy as np
import logging
import sys
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from unet_model import UNet
from satelliteLoader import satelliteDataSet

batch_size = 1
num_workers = 4
valid_size = 0.1
shuffle = True
log_steps = 25
metrics_debug = True

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(1024),
        transforms.RandomHorizontalFlip(),
        #transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        #transforms.CenterCrop(1024),
        #transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../data'
logdir = '../logs'


# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
#train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
#val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['val'])
train_dataset = satelliteDataSet(os.path.join(data_dir, 'train'), data_transforms['train'])
val_dataset = satelliteDataSet(os.path.join(data_dir, 'train'), data_transforms['val'])


num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(0)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, sampler=train_sampler,
    num_workers=num_workers
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, sampler=valid_sampler,
    num_workers=num_workers
)

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
dataloaders = {
    'train': train_loader,
    'val': val_loader,
}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
}


def initializeLogging(log_filename, logger_name):
  log = logging.getLogger(logger_name)
  log.setLevel(logging.DEBUG)
  log.addHandler(logging.StreamHandler(sys.stdout))
  log.addHandler(logging.FileHandler(log_filename, mode='a'))

  return log

'''
def save_checkpoint(state, is_best, checkpoint_folder=logdir,
                filename='checkpoint.pth.tar'):
  filename = os.path.join(checkpoint_folder, filename)
  best_model_filename = os.path.join(checkpoint_folder, 'model_best.pth.tar')
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, best_model_filename)
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # logging setup
    logger_name = 'train_logger'
    logger = initializeLogging(os.path.join(logdir,
                'train_history.txt'), logger_name)

    #  checkpointing setup
    checkpoint_frequency = log_steps
    logger.info('started model training')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print(phase + ' size: ', str(dataset_sizes[phase]))
            # Iterate over data
            for inputs,labels in tqdm(dataloaders[phase]):
                inputs = inputs.permute(0,3,1,2)
                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.softmax(outputs, 1)
                    preds = preds.squeeze(0).permute(1,2,0)
                    dist = torch.distributions.Categorical(preds)
                    preds = dist.sample()
                    preds = preds.unsqueeze(0)
                    loss = criterion(outputs.float(), labels.long())
                    preds = preds.float()#.cpu()
                    #labels = labels.cpu()
                    running_corrects += preds.eq(labels.view_as(preds)).sum().item()
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()     # back propagation
                        optimizer.step()    # actually update weights/parameters
                #print(running_corrects, loss)
                # statistics
                # running_corrects += torch.sum(preds == labels.long().data)
                # running_num_data += labels[labels >= 0].size(0)
                running_loss += loss.item()
                #running_loss += loss.item()
                #running_corrects += torch.sum(preds[labels >= 0] == labels[labels >= 0].data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / (dataset_sizes[phase]*1024*1024)
            if epoch_acc > best_acc:
                checkpoint_dict = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': epoch_acc,
                        'epoch_loss': epoch_loss
                        }
                best_acc = epoch_acc
                torch.save(checkpoint_dict, 'best_model.pt')
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc
            ))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_second_layer(model, criterion, optimizer, scheduler, num_epochs=25):
    model1 = UNet()
    model2 = UNet()
    model1.load_state_dict(best_model_wts)
    model2.load_state_dict(best_model_wts)

if __name__ == '__main__':
    lr = 0.0001
    model = UNet()
    #num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2
    # Alternatively it can be generalized to nn.Linear(num_ftrs, len(class_names))
    #model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=lr)#, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Finetune training the convnet and evaluation
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

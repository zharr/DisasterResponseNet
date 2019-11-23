import numpy as np
import os
import schedulers
from statistics import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import unittest
from unet_model import UNet
from satelliteLoader import satelliteDataSet
from torch.utils.data.sampler import SubsetRandomSampler

lr = 0.0001
lr_scheduler_divide_every_n_epochs = 20
lr_scheduler_type = 'WarmupAndExponentialDecayScheduler'
lr_scheduler_divisor = 5
momentum=None
num_cores = 8
batch_size = 64
test_set_batch_size = 128
num_epochs = 10
num_workers = 64
valid_size = 0.1
shuffle = True
log_steps = 2
datadir = '../data'

def train_unet():
  print('==> Preparing data..')
  img_dim = 1024



  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dataset = satelliteDataSet(
      os.path.join(datadir, 'train'),
      transforms.Compose([
        transforms.RandomResizedCrop(img_dim),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
      ]))
  train_dataset_len = len(train_dataset)

  resize_dim = max(img_dim, 256)
  test_dataset = satelliteDataSet(
      os.path.join(datadir, 'train'),
      transforms.Compose([
          transforms.Resize(resize_dim),
          transforms.CenterCrop(img_dim),
          transforms.ToTensor(),
          normalize,
      ]))

  train_sampler = None
  test_sampler = None
  if xm.xrt_world_size() > 1:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=batch_size,
      sampler=train_sampler,
      shuffle=False if train_sampler else True,
      num_workers=num_workers)
  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=test_set_batch_size,
      sampler=test_sampler,
      shuffle=False,
      num_workers=num_workers)

  torch.manual_seed(42)

  devices = (xm.get_xla_supported_devices(max_devices=num_cores))

  torchvision_model = UNet()
  model_parallel = dp.DataParallel(torchvision_model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = context.getattr_or(
        'optimizer', lambda: optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=1e-4))
    lr_scheduler = context.getattr_or(
        'lr_scheduler', lambda: schedulers.wrap_optimizer_with_scheduler(
            optimizer,
            scheduler_type=lr_scheduler_type,
            scheduler_divisor=lr_scheduler_divisor,
            scheduler_divide_every_n_epochs=lr_scheduler_divide_every_n_epochs,
            num_steps_per_epoch=num_training_steps_per_epoch,
            summary_writer=None))
    tracker = xm.RateTracker()
    model.train()
    for x, (data, target) in loader:
      optimizer.zero_grad()
      data = data.permute(0,3,1,2)
      output = model(data)
      print('passed through model')
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(batch_size)
      if x % log_steps == 0:
        print('device: {}, x: {}, loss: {}, tracker: {}, tracker_global: {} '.format(device, x, loss.item(),
                                                                                     tracker.rate(), tracker.global_rate()))
      if lr_scheduler:
        lr_scheduler.step()

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    model.eval()
    for x, (data, target) in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct / total_samples
    #test_utils.print_test_update(device, accuracy)
    print('device: {}, accuracy: {}'.format(device, accuracy))
    return accuracy

  accuracy = 0.0
  num_devices = len(
      xm.xla_replication_devices(devices)) if len(devices) > 1 else 1
  num_training_steps_per_epoch = train_dataset_len // (
      batch_size * num_devices)
  for epoch in range(1, num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = mean(accuracies)
    print('Epoch: {}, Mean Accuracy: {:.2f}%'.format(epoch, accuracy))
    global_step = (epoch - 1) * num_training_steps_per_epoch
    print('global step: {}'.format(global_step)

  return accuracy




torch.set_default_tensor_type('torch.FloatTensor')
train_unet()

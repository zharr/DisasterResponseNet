import os
from statistics import mean
import shutil
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.debug.metrics as met
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
from sklearn.metrics import confusion_matrix
import test_utils
from unet_model import UNet
from satelliteLoader import satelliteDataSet

datadir = '../data'
logdir = '../logs'
batch_size = 2
momentum = 0.99
num_epochs = 18
num_cores = 8
num_workers = 2
drop_last = True
log_steps = 25
metrics_debug = True
num_classes = 4

def initializeLogging(log_filename, logger_name):
  log = logging.getLogger(logger_name)
  log.setLevel(logging.DEBUG)
  log.addHandler(logging.StreamHandler(sys.stdout))
  log.addHandler(logging.FileHandler(log_filename, mode='a'))

  return log

def save_checkpoint(state, is_best, checkpoint_folder=logdir,
                filename='checkpoint.pth.tar'):
  filename = os.path.join(checkpoint_folder, filename)
  best_model_filename = os.path.join(checkpoint_folder, 'model_best.pth.tar')
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, best_model_filename)

def train_unet():

  # logging setup
  logger_name = 'train_logger'
  logger = initializeLogging(os.path.join(logdir,
                'train_history.txt'), logger_name)

  # checkpointing setup
  checkpoint_frequency = log_steps
  last_checkpoint = 0


  torch.manual_seed(1)

  '''
  train_dataset = datasets.MNIST(
      datadir,
      train=True,
      download=True,
      transform=transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))]))
  train_dataset_len = len(train_dataset)
  test_dataset = datasets.MNIST(
      datadir,
      train=False,
      transform=transforms.Compose(
          [transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))]))
  '''

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
      drop_last=drop_last,
      shuffle=False if train_sampler else True,
      num_workers=num_workers)
  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=batch_size,
      sampler=test_sampler,
      drop_last=drop_last,
      shuffle=False,
      num_workers=num_workers)

  devices = (
      xm.get_xla_supported_devices(
          max_devices=num_cores) if num_cores != 0 else [])
  # Scale learning rate to num cores
  lr = 0.01
  lr = lr * max(len(devices), 1)
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  model_parallel = dp.DataParallel(UNet, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = context.getattr_or(
        'optimizer',
        lambda: optim.Adam(model.parameters(), lr=lr))
    tracker = xm.RateTracker()

    model.train()
    maxItr = len(loader)
    print('# of iterations: {}'.format(maxItr))
    logger.info('# of iterations: {}'.format(maxItr))
    for x, (data, target) in enumerate(loader):
      data = target[0].permute(0,3,1,2)
      target = target[1]
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target.long())
      #_, preds = torch.max(output, 1)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(batch_size)

      # compute the confusion matrix and IoU
      val_conf = np.zeros((num_classes, num_classes))
      val_conf = val_conf + confusion_matrix(
          target[target >= 0].view(-1).cpu().numpy(),
          output[target >= 0].view(-1).cpu().numpy())
      pos = np.sum(val_conf, 1)
      res = np.sum(val_conf, 0)
      tp = np.diag(val_conf)
      iou = np.mean(tp / np.maximum(1, pos + res - tp))

      if x % log_steps == 0:
        print('device: {}, x: {}, loss: {}, tracker_rate: {}, tracker_global_rate: {}'.format(device, x, loss.item(), tracker.rate(), tracker.global_rate()))
        logger.info('device: {}, x: {}, loss: {}, tracker_rate: {}, tracker_global_rate: {}'.format(device, x, loss.item(), tracker.rate(), tracker.global_rate()))
        test_utils.print_training_update(device, x, loss.item(),
                                         tracker.rate(),
                                         tracker.global_rate())

      if x % checkpoint_frequency == 0 or x == maxItr - 1:
          is_best = iou > best_iou
          if is_best:
              best_iou = iou

          do_checkpoint = (x - last_checkpoint) >= checkpoint_frequency
          if is_best or x == maxItr - 1 or do_checkpoint:
              last_checkpoint = x
              checkpoint_dict = {
                  'itr': x + 1,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'best_iou': best_iou
              }
              save_checkpoint(checkpoint_dict, is_best, checkpoint_folder=logdir)

  def test_loop_fn(model, loader, device, context):
    total_samples = 0
    correct = 0
    model.eval()
    for data, target in loader:
      output = model(data)
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()
      total_samples += data.size()[0]

    accuracy = 100.0 * correct / total_samples
    test_utils.print_test_update(device, accuracy)
    logger.info('TEST: device: {}, accuracy: {}'.format(device, accuracy))
    return accuracy

  accuracy = 0.0
  writer = test_utils.get_summary_writer(logdir)
  num_devices = len(
      xm.xla_replication_devices(devices)) if len(devices) > 1 else 1
  num_training_steps_per_epoch = train_dataset_len // (
      batch_size * num_devices)
  print('total epochs: {}'.format(num_epochs))
  for epoch in range(1, num_epochs + 1):
    print(epoch)
    print(train_loader)
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, test_loader)
    accuracy = mean(accuracies)
    print('Epoch: {}, Mean Accuracy: {:.2f}%'.format(epoch, accuracy))
    logger.info('Epoch: {}, Mean Accuracy: {:.2f}%'.format(epoch, accuracy))
    global_step = (epoch - 1) * num_training_steps_per_epoch
    test_utils.add_scalar_to_summary(writer, 'Accuracy/test', accuracy,
                                     global_step)
    if metrics_debug:
      print(met.metrics_report())
      logger.info(met.metrics_report())

  test_utils.close_summary_writer(writer)
  return accuracy



# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
train_unet()

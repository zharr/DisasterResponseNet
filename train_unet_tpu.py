import os
from statistics import mean
import shutil
import time
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
import torch_xla.test.test_utils as test_utils
from unet_model import UNet
from satelliteLoader import satelliteDataSet

datadir = '../data'
logdir = '../logs'
batch_size = 128
momentum = 0.5
lr = 0.01
num_epochs = 18
num_cores = 8
num_workers = 12
drop_last = True
log_steps = 10
metrics_debug = True


def train_unet():
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
  lr = lr * max(len(devices), 1)
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  model_parallel = dp.DataParallel(UNet, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.NLLLoss()
    optimizer = context.getattr_or(
        'optimizer',
        lambda: optim.SGD(model.parameters(), lr=lr, momentum=momentum))
    tracker = xm.RateTracker()

    model.train()
    for x, (data, target) in enumerate(loader):
      optimizer.zero_grad()
      output = model(data)
      loss = loss_fn(output, target)
      loss.backward()
      xm.optimizer_step(optimizer)
      tracker.add(batch_size)
      if x % log_steps == 0:
        test_utils.print_training_update(device, x, loss.item(),
                                         tracker.rate(),
                                         tracker.global_rate())

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
    return accuracy

  accuracy = 0.0
  writer = test_utils.get_summary_writer(logdir)
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
    test_utils.add_scalar_to_summary(writer, 'Accuracy/test', accuracy,
                                     global_step)
    if metrics_debug:
      print(met.metrics_report())

  test_utils.close_summary_writer(writer)
  return accuracy



# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
train_unet()

import args_parse

SUPPORTED_MODELS = [
    'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
    'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
    'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13',
    'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn'
]

MODEL_OPTS = {
    '--model': {
        'choices': SUPPORTED_MODELS,
        'default': 'resnet50',
    },
    '--test_set_batch_size': {
        'type': int,
    },
    '--lr_scheduler_type': {
        'type': str,
    },
    '--lr_scheduler_divide_every_n_epochs': {
        'type': int,
    },
    '--lr_scheduler_divisor': {
        'type': int,
    },
}

FLAGS = args_parse.parse_common_options(
    datadir='../data',
    batch_size=128,
    num_epochs=10,
    momentum=None,
    lr=0.0001,
    target_accuracy=None,
    opts=MODEL_OPTS.items(),
)
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

DEFAULT_KWARGS = dict(
    batch_size=128,
    test_set_batch_size=64,
    num_epochs=18,
    momentum=0.9,
    lr=0.1,
    target_accuracy=0.0,
)
MODEL_SPECIFIC_DEFAULTS = {
    # Override some of the args in DEFAULT_KWARGS, or add them to the dict
    # if they don't exist.
    'resnet50':
        dict(
            DEFAULT_KWARGS, **{
                'lr': 0.8,
                'lr_scheduler_divide_every_n_epochs': 20,
                'lr_scheduler_divisor': 5,
                'lr_scheduler_type': 'WarmupAndExponentialDecayScheduler',
            })
}

# Set any args that were not explicitly given by the user.
default_value_dict = MODEL_SPECIFIC_DEFAULTS.get(FLAGS.model, DEFAULT_KWARGS)
for arg, value in default_value_dict.items():
  if getattr(FLAGS, arg) is None:
    setattr(FLAGS, arg, value)

MODEL_PROPERTIES = {
    'inception_v3': {
        'img_dim': 299,
        'model_fn': lambda: torchvision.models.inception_v3(aux_logits=False)
    },
    'DEFAULT': {
        'img_dim': 224,
        'model_fn': getattr(torchvision.models, FLAGS.model)
    }
}


def get_model_property(key):
  return MODEL_PROPERTIES.get(FLAGS.model, MODEL_PROPERTIES['DEFAULT'])[key]


def train_imagenet():
  print('==> Preparing data..')
  '''
  img_dim = get_model_property('img_dim')
  if FLAGS.fake_data:
    train_dataset_len = 1200000  # Roughly the size of Imagenet dataset.
    train_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
        sample_count=train_dataset_len // FLAGS.batch_size //
        xm.xrt_world_size())
    test_loader = xu.SampleGenerator(
        data=(torch.zeros(FLAGS.test_set_batch_size, 3, img_dim, img_dim),
              torch.zeros(FLAGS.test_set_batch_size, dtype=torch.int64)),
        sample_count=50000 // FLAGS.batch_size // xm.xrt_world_size())
  else:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(img_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_len = len(train_dataset.imgs)
    resize_dim = max(img_dim, 256)
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.datadir, 'val'),
        # Matches Torchvision's eval transforms except Torchvision uses size
        # 256 resize for all models both here and in the train loader. Their
        # version crashes during training on 299x299 images, e.g. inception.
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
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=FLAGS.test_set_batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=FLAGS.num_workers)
  '''
  batch_size = 8
  num_workers = 64
  valid_size = 0.1
  shuffle = True

  data_transforms = {
    'train': transforms.Compose([
      transforms.RandomResizedCrop(256),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
          # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
          # transforms.Resize(256),
      transforms.RandomResizedCrop(256),
      transforms.ToTensor(),
          # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  }

  data_dir = '../data'

# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
#train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
#val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['val'])
  train_dataset = satelliteDataSet(os.path.join(data_dir, 'train'), data_transforms['train'])
  val_dataset = satelliteDataSet(os.path.join(data_dir, 'train'), data_transforms['val'])

  
  train_dataset_len = len(train_dataset)
  test_dataset_len = len(val_dataset)
  indices = list(range(train_dataset_len))
  split = int(np.floor(valid_size * train_dataset_len))

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


  torch.manual_seed(42)

  devices = (
      xm.get_xla_supported_devices(
          max_devices=FLAGS.num_cores) if FLAGS.num_cores != 0 else [])
  # Pass [] as device_ids to run using the PyTorch/CPU engine.
  #torchvision_model = get_model_property('model_fn')
  torchvision_model = UNet()
  model_parallel = dp.DataParallel(torchvision_model, device_ids=devices)

  def train_loop_fn(model, loader, device, context):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = context.getattr_or(
        'optimizer', lambda: optim.SGD(
            model.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
            weight_decay=1e-4))
    lr_scheduler = context.getattr_or(
        'lr_scheduler', lambda: schedulers.wrap_optimizer_with_scheduler(
            optimizer,
            scheduler_type=getattr(FLAGS, 'lr_scheduler_type', None),
            scheduler_divisor=getattr(FLAGS, 'lr_scheduler_divisor', None),
            scheduler_divide_every_n_epochs=getattr(
                FLAGS, 'lr_scheduler_divide_every_n_epochs', None),
            num_steps_per_epoch=num_training_steps_per_epoch,
            summary_writer=writer if xm.is_master_ordinal() else None))
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
      tracker.add(FLAGS.batch_size)
      if x % FLAGS.log_steps == 0:
        test_utils.print_training_update(device, x, loss.item(), tracker.rate(),
                                         tracker.global_rate())
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
    test_utils.print_test_update(device, accuracy)
    return accuracy

  accuracy = 0.0
  writer = SummaryWriter(log_dir=FLAGS.logdir) if FLAGS.logdir else None
  num_devices = len(
      xm.xla_replication_devices(devices)) if len(devices) > 1 else 1
  num_training_steps_per_epoch = train_dataset_len // (
      FLAGS.batch_size * num_devices)
  for epoch in range(1, FLAGS.num_epochs + 1):
    model_parallel(train_loop_fn, train_loader)
    accuracies = model_parallel(test_loop_fn, val_loader)
    accuracy = mean(accuracies)
    print('Epoch: {}, Mean Accuracy: {:.2f}%'.format(epoch, accuracy))
    global_step = (epoch - 1) * num_training_steps_per_epoch
    test_utils.add_scalar_to_summary(writer, 'Accuracy/test', accuracy,
                                     global_step)
    if FLAGS.metrics_debug:
      print(met.metrics_report())

  return accuracy




# Run the tests.
torch.set_default_tensor_type('torch.FloatTensor')
#run_tests()
train_imagenet()

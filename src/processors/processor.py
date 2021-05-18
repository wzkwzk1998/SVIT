import os
import sys
import pickle
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import tensorboard
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from collections import OrderedDict
from feeder.feeder import Feeder
from datetime import datetime


########################  tool function #########################
def import_class(name):
    """
        a function use to dynamically load model
    :param name:
    :return:
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


##################### processor ##########################
class Processor():
    """
        Processor for Skeleton-based recognition
    """

    def __init__(self, arg=None):
        self.load_arg(arg)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

    def load_arg(self, arg):
        parser = self.get_parser()

        p = parser.parse_args(arg)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as config_file:
                default_arg = yaml.load(config_file, Loader=yaml.FullLoader)

            key = vars(p).keys()
            for k in default_arg.keys():
                assert k in key

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args()

    def init_environment(self):
        if not self.arg.debug:
            self.writer = SummaryWriter('../log/{}'.format(datetime.now().strftime('%b%d_%H-%M-%S')))
        self.epoch_info = dict()

    def load_model(self):
        print('[ use Model ]: {}'.format(self.arg.model))
        Model = import_class(self.arg.model)
        model = Model(**self.arg.model_args)
        loss = nn.CrossEntropyLoss()
        self.model = model
        self.loss = loss
        self.param_total = sum([param.nelement() for param in self.model.parameters()])
        print(self.model)
        print("[Number of parameter]: %.2fM" % (self.param_total / 1e6))

    def load_weights(self):
        if self.arg.debug:
            return
        if self.arg.weights_path is None:
            return
        ignore_weights = self.arg.ignore_weights
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        print('Load weights from {}.'.format(self.arg.weights_path))
        weights = torch.load(self.arg.weights_path)
        weights = OrderedDict([[k.split('module.')[-1],
                                v.cpu()] for k, v in weights.items()])

        # filter weights
        for i in ignore_weights:
            ignore_name = list()
            for w in weights:
                if w.find(i) == 0:
                    ignore_name.append(w)
            for n in ignore_name:
                weights.pop(n)
                print('Filter [{}] remove weights [{}].'.format(i,n))

        for w in weights:
            print('Load weights [{}].'.format(w))

        try:
            self.model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = self.model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for d in diff:
                print('Can not find weights [{}].'.format(d))
            state.update(weights)
            self.model.load_state_dict(state)

    def gpu(self):
        if self.arg.use_gpu:
            print('[ using gpu ]')
            self.gpus = [self.arg.device] if isinstance(self.arg.device, int) else list(self.arg.device)
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, self.gpus)))
            self.dev = 'cuda:0'
        else:
            print('[ using cpu ]')
            self.dev = 'cpu'

        self.model = self.model.to(self.dev)
        if self.arg.use_gpu and len(self.gpus) > 1 and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def load_data(self):
        """
        load skeleton data for train.yaml and test
        :return: train.yaml and test dataloader as a dict
        """
        print('[ use feeder ]: {}'.format(self.arg.feeder))
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.train_feeder_args:
            self.data_loader['train'] = DataLoader(Feeder(**self.arg.train_feeder_args,
                                                          debug=self.arg.debug),
                                                   batch_size=self.arg.train_batchsize,
                                                   shuffle=True,
                                                   num_workers=self.arg.num_workers)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = DataLoader(Feeder(**self.arg.test_feeder_args,
                                                         debug=self.arg.debug),
                                                  batch_size=self.arg.test_batchsize,
                                                  shuffle=False,
                                                  num_workers=self.arg.num_workers)

    def load_optimizer(self):
        print('[use optimizer] : {}'.format(self.arg.optimizer))
        print('[learning rate is] : {}'.format(self.arg.base_lr))
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       weight_decay=self.arg.weight_decay,
                                       momentum=self.arg.momentum
                                       )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        if self.arg.lr_scheduler == 'Step':
            self.lr_scheduler = lr_scheduler.StepLR(optimizer=self.optimizer,
                                                    step_size=self.arg.step_size,
                                                    gamma=self.arg.scheduler_gamma)
        elif self.arg.lr_scheduler == 'Cos':
            self.lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                               T_max=self.arg.T_max)

    def train(self):
        """
        train one epoch
        :return:
        """
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []
        for data, label in loader:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record
            loss_value.append(loss.data.item())

        self.lr_scheduler.step()
        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        print('train epoch mean is {}'.format(self.epoch_info['train_mean_loss']))

    def test(self):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        # eval
        for data, label in loader:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            loss = self.loss(output, label)
            loss_value.append(loss.data.item())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        self.label = np.reshape(self.label, (self.label.shape[0]))

        self.epoch_info['test_mean_loss'] = np.mean(loss_value)
        print('test epoch mean loss is {}'.format(self.epoch_info['test_mean_loss']))

        # show acc
        print('==================================================')
        self.epoch_info['acc'] = {}
        for k in self.arg.show_topk:
            self.epoch_info['acc']['top-{}'.format(k)] = self.show_topk(k)

    def start(self):
        self.epoch_info['epoch_num'] = 0

        if self.arg.phase == 'train':
            self.epoch_info['max_acc'] = 0.0
            for epoch in range(0, self.arg.num_epoch):
                print('epoch is {}'.format(epoch))
                self.epoch_info['epoch_num'] += 1
                # train
                self.train()
                if not self.arg.debug:
                    self.writer.add_scalar('data/train_loss', self.epoch_info['train_mean_loss'], self.epoch_info['epoch_num'])


                # eval
                if ((epoch + 1) % self.arg.test_interval == 0) or (
                        (epoch + 1) % self.arg.num_epoch == 0):
                    self.test()
                    if not self.arg.debug:
                        self.writer.add_scalar('data/test_loss', self.epoch_info['test_mean_loss'], self.epoch_info['epoch_num'])
                        self.writer.add_scalars('data/top-1', self.epoch_info['acc'], self.epoch_info['epoch_num'])
                    if(self.epoch_info['acc']['top-1'] > self.epoch_info['max_acc']):
                        self.epoch_info['max_acc'] = self.epoch_info['acc']['top-1']
                        if not self.arg.debug:
                            self.save_model(self.arg.weights_save_path)


        elif self.arg.phase == 'eval':
            self.test()
        else:
            raise ValueError()

        if not self.arg.debug:
            self.writer.close()

    def show_topk(self, k):
        rank = self.result.argsort()  # the index of array from small to large
        top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        acc = sum(top_k) * 1.0 / len(top_k)
        print('Top-{} acc: {:.2f}%'.format(k, 100 * acc))
        return acc

    def save_model(self, model_path):
        model_dir = os.path.join(*(model_path.split(os.sep)[0:-1]))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        state_dict = self.model.state_dict()
        weights = OrderedDict([[''.join(k.split('module.')),
                                v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, model_path)
        print('The model has been saved as {}.'.format(model_path))

    @staticmethod
    def get_parser():

        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=False, description='Param for recognition')

        parser.add_argument('-c', '--config', default='../config/svit/ntu-xview/train.yaml',
                            help='path to configuration file')

        # Processor argument
        parser.add_argument('--num_epoch', type=int, default=0, help='training epoch')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')
        parser.add_argument('--phase', default='train', help='choose train or test')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--test_interval', type=int, default=5, help='')
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which top-k acc is evaluated')
        parser.add_argument('--debug', type=str2bool, default=False, help='use debug mode')

        # optimizer
        parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
        parser.add_argument('--weight_decay', type=int, default=0.0001, help='weight decay of the optimizer')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9)

        # lr_scheduler
        parser.add_argument('--lr_scheduler', default='Cos', help='learning rate scheduler used')
        parser.add_argument('--T_max', type=int, default=3)
        parser.add_argument('--step_size', type=int, default=100)
        parser.add_argument('--scheduler_gamma', type=int, default=0.1)


        # DataLoader
        parser.add_argument('--num_workers', type=int, default=2, help='num_workers for dataloader')
        parser.add_argument('--train_batchsize', type=int, default=64, help='batchsize for train')
        parser.add_argument('--test_batchsize', type=int, default=64, help='batchsize for test')

        # model argument
        parser.add_argument('--model', default='net.svit.Model', help='the model will be used')
        parser.add_argument('--model_args', default=dict(), help='model args')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder.Feeder', help='feeder used for dataloader')
        parser.add_argument('--train_feeder_args', default=dict(), help='train feeder_args')
        parser.add_argument('--test_feeder_args', default=dict(), help='test feeder_args')

        # weight
        parser.add_argument('--weights_path', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--weights_save_path', default=None, help='the weight file path for model')

        return parser

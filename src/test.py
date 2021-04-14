import torch
import torch.nn as nn
from processors import processor
from net.svit import MLPBlock
from net.svit import Encoder
from net.svit import Model
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter()
    x = torch.randn((20, 3, 300, 25, 2))
    model = Model(num_class=60,
                  num_layer=4,
                  mlp_dim=x.shape[3]*x.shape[1],
                  hidden=64)
    writer.add_graph(model=model,
                            input_to_model=x)
    writer.close()

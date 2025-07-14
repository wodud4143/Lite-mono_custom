from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from trainer import Trainer
from torch.autograd import profiler

options = LiteMonoOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
    

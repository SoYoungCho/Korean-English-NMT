import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_loader
from simple_nmt.search import SingleBeamSearchSpace



class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()
        print("==== generator init =====")
        self.output = nn.Linear(hidden_size, output_size) ##
        self.softmax = nn.LogSoftmax(dim=-1) ##

    def forward(self, x):
        # |x| = (batch_size, length, hidden_size)
        print("==== generator forward =====")
        y = self.softmax(self.output(x))
        # |y| = (batch_size, length, output_size)

        # Return log-probability instead of just probability.
        print("==== generator forward end =====")
        return y
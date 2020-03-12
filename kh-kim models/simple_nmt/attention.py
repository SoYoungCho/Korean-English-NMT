import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_loader
from simple_nmt.search import SingleBeamSearchSpace


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        print("==== attention init =====")
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        print("==== attention forward =====")
        # |h_src| = (batch_size, length, hidden_size)
        # |h_t_tgt| = (batch_size, 1, hidden_size)
        # |mask| = (batch_size, length)

        query = self.linear(h_t_tgt.squeeze(1)).unsqueeze(-1) # 1을 쥐어짜 없앰. -1은 맨 오른쪽 인덱스 자리에 해당하는 차원을 만드어줌.
        # |query| = (batch_size, hidden_size, 1)

        weight = torch.bmm(h_src, query).squeeze(-1) # batch matrix multiplication
        # |weight| = (batch_size, length)
        if mask is not None:
            # Set each weight as -inf, if the mask value equals to 1.
            # Since the softmax operation makes -inf to 0,
            # masked weights would be set to 0 after softmax operation.
            # Thus, if the sample is shorter than other samples in mini-batch,
            # the weight for empty time-step would be set to 0.

            #weight.masked_fill_(mask, -float('inf'))
            weight.masked_fill_(mask, False) # warning 안나게 하려면 이렇게 해야하나
        weight = self.softmax(weight)

        context_vector = torch.bmm(weight.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)
        print("==== attention forward end =====")
        return context_vector
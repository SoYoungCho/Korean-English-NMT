import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import data_loader
from simple_nmt.search import SingleBeamSearchSpace


class Encoder(nn.Module):

    def __init__(self, word_vec_dim, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()
        print("====encoder init=====")
        # Also, its hidden_size is half of original hidden_size,
        # because it is bidirectional.
        self.rnn = nn.LSTM(word_vec_dim, # 다른 코드에서는 feature size로 많이 표현. 몇 차원으로 해줄 건지. 이건 우리가 정해줘야하는 값 (예) 512)
                           int(hidden_size / 2),
                           num_layers=n_layers,
                           dropout=dropout_p,
                           bidirectional=True,
                           batch_first=True # 형상에서 '배치'를 먼저 주겠다는 뜻
                           )

    def forward(self, emb): # batch 1개 기준
        # |emb| = (batch_size, length, word_vec_dim)
        print("====encoder forward=====")

        if isinstance(emb, tuple): # emb가 tuple이 맞으면 true
            x, lengths = emb # 여기서 어떻게 되는지 모르겠음 .. batch_size X word_vec_dim 이게 x에 들어가고 lengths에 length가 들어가는듯..?
            x = pack(x, lengths.tolist(), batch_first=True, enforce_sorted=False)

            # Below is how pack_padded_sequence works.
            # As you can see,
            # PackedSequence object has information about mini-batch-wise information,
            # not time-step-wise information.
            #
            # 만약 토큰 단위일 경우 [['안녕', '만나서', '반가워'],['반가워', '친구']]
            # 이를 텐서로 바꾸면 [torch.tensor([1,2,3]), torch.tensor([3,4])]라 하자
            # pad sequence 하면 가장 긴 텐서 길이 기준 0으로 패딩 (행 단위, axis = 1이라고 생각하자)
            # pack_padded_sequence( 패딩된 시퀀스 pad sequence, 각 시퀀스 길이(즉, 각 문장의 길이) ) 넘겨주면
            # 열 단위로 묶어서 1차원으로 축소 (axis = 0으로 flatten화) 한다고 생각하면 된다.
            # 이 때 그 axis =0 기준으로 있는 인덱스 길이도 함께 가지고 있다. (아래 코드에서 batch_sizes) 이렇게 가지고 있어야 이 길이 기준으로 잘라서 다시 shape.
            # 위 영어 설명 중 mini-batch-wise info를 가지고 있다는 건 위 내용

            # a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
            # b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
            # >>>>
            # tensor([[ 1,  2,  3],
            #     [ 3,  4,  0]])

            # torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2]
            # >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))

        else: # emb가 tuple 타입이 아님
            x = emb

        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers * 2, batch_size, hidden_size / 2)

        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)
        print("====encoder forward end=====")
        return y, h
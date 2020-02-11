import torch.nn as nn

class EncoderRNN(nn.Module):
    """
    인코더 같은 경우는 사실 그냥 LSTM (GRU or RNN) 에 입력을 넣어주고 해당 결과를 리턴만 하면 됩니다.
    IBM 코드 같은 경우는 이것저것 고려하여 코드가 길지만, 실질적으로 사용하는 부분은 아래 내용이 대부분이에요.
    필요하면 하나씩 추가하면 됩니다.

    예를 들어 Convolution Layer를 추가적으로 넣으신다면 아래 링크가면 제가 적용한 코드가 있습니다.
    https://github.com/sh951011/Korean-Speech-Recognition/blob/master/models/listener.py
    아래 내용을 바탕으로 조금씩 추가한 코드일 뿐 아래 코드와 크게 다르지 않습니다.
    """
    def __init__(self, feat_size, hidden_size, dropout_p=0.5, layer_size=5, bidirectional=True, rnn_cell='gru'):
        super(EncoderRNN, self).__init__()
        # rnn_cell을 LSTM, GRU, RNN 중 어떤 것을 할지 정해주는 부분입니다.
        # nn은 Neural Network 줄임말로, torch에서 제공하는 유명한 Neural Network 클래스들을 제공합니다.
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        # 정한 rnn_cell을 기반으로 하여, rnn에 넣을 입력 벡터 사이즈와, 히든 스테이트 사이즈, 레이어 사이즈, Bidirectional 여부 등
        # 필요한 정보를 넣어줍니다.
        # batch_first 옵션 같은 경우는 입력 데이터의 형상이 BxS (B : batch size, S : sequence length) 와 같이 배치가 먼저 오는 경우 True,
        # 배치가 중간에 오는 경우는 batch_first를 False로 주면 됩니다.
        # 입력 데이터를 넣는 방법도 사람마다 다르다보니, 여러 방법으로 가능하도록 넣은 기능이에요.
        self.rnn = self.rnn_cell(input_size=feat_size, hidden_size=hidden_size, num_layers=layer_size,
                                 bias=True, batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, inputs):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            inputs (batch, seq_len): tensor containing the features of the input sequence.

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
                          => Ex (32, 257, 512)
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
                          => Ex (16, 32, 512)
        """
        # 이쪽 부분은 inputs을 어떤 형식으로 넣을지를 몰라서
        # 일단 기본적으로만 해놨습니다.
        # 차원이 안 맞으면, squeeze()나 unsqueeze()를 추가적으로 해주시면 될거에요.
        # squeeze() & unsqueeze()의 간단한 설명은 디코더쪽에 적었어요
        x = inputs

        if self.training:
            self.rnn.flatten_parameters()
        # nn.RNN, nn.LSTM, nn.GRU 중 선택한 RNN의 forward()를 진행합니다.
        # 아래와 같이 호출만 해주면 내부적으로 알아서 처리해주기 때문에 편리합니다.
        # RNN은 2가지를 반환하는데, outputs는 RNN이 각 타입스텝마다 내놓은 output들을 모아놓은 것입니다.
        # hiddens는 사용한 RNN의 Hidden State입니다. 
        outputs, hiddens = self.rnn(x)

        return outputs, hiddens

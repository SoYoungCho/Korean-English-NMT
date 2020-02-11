import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(nn.Module):
     def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 layer_size=1, rnn_cell='gru', dropout_p=0,
                 use_attention=True, device=None, use_beam_search=True, k=8):
        super(DecoderRNN, self).__init__()
        self.rnn_cell = nn.LSTM if rnn_cell.lower() == 'lstm' else nn.GRU if rnn_cell.lower() == 'gru' else nn.RNN
        self.rnn = self.rnn_cell(hidden_size , hidden_size, layer_size, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.layer_size = layer_size
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.device = device
        if use_attention:
            self.attention = Attention(self.hidden_size)

    def _forward_step(self, decoder_input, decoder_hidden, encoder_outputs, function):
        """  디코더에서 한 타입 스텝마다 forwarding 해주는 함수 """
        batch_size = decoder_input.size(0)
        output_size = decoder_input.size(1)
        embedded = self.embedding(decoder_input)
        embedded = self.input_dropout(embedded)
        # rnn의 파라미터를 내부적으로 일자로 펴주는 함수입니다.
        # 훈련시 펴주고 시작하면 더 빠르다고 합니다.
        if self.training:
            self.rnn.flatten_parameters()
        # 현재 타입 스텝에서의 디코더의 output
        decoder_output, hidden = self.rnn(embedded, decoder_hidden) # decoder output

        # 어텐션 적용
        if self.use_attention:
            output = self.attention(decoder_output=decoder_output, encoder_output=encoder_outputs)
        else: output = decoder_output
        # torch.view()에서 -1이면 나머지 알아서 맞춰줌
        # torch.view() 같은 경우는 tensor의 형상을 조절할 수 있는 함수입니다.
        # 어떤 텐서의 모양이 (32, 3, 6) 이였는데 이를 (32, 3 * 6) 으로 바꿔주고 싶다면 tensor.view(32, 3 * 6)으로 바꿔주면 됩니다.
        # 반대로, 이 (32, 18) 을 다시 (32, 3, 6) 으로 재배열 해주고 싶다면, tensor.view(32, 3, 6) 와 같이 사용하면 됩니다.
        # torch.view() 같은 경우는 numpy에서의 np.reshape()에 해당하는 함수라고 생각하면 돼요.
        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        # softmax로 먹여서 총 classfication 에 대한 확률 분포를 리턴합니다.

        return predicted_softmax

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax, teacher_forcing_ratio=0.99):
        """
        디코더의 전체 forward()
        여기서는 for문을 이용해서 전체 타입 스텝에 대해 디코딩을 하거나,
        티쳐포싱 사용시, 병렬적으로 처리합니다. (티쳐포싱 사용시, 이미 있는 레이블을 입력으로 넣어주므로 순서대로 할 필요가 없습니다.)
        반대로, 티쳐포싱 미사용시에는 이전 타임 스텝의 output이 input으로 들어오게 되므로, 병렬적으로 처리하지 못하고 순서대로 처리해야합니다.
        """
        decode_results = []
        # Validate Arguments
        batch_size = inputs.size(0)
        max_length = inputs.size(1) - 1  # minus the start of sequence symbol
        # Initiate decoder Hidden State to zeros  :  LxBxH
        # 기존 IBM Pytorch-seq2seq는 인코더와 디코더의 레이어 사이즈가 같아야만 돌아갑니다.
        # IBM은 구현시에, 인코더의 모든 레이어의 히든스테이트로 디코더의 히든 스테이트를 초기화합니다.
        # 하지만 대부분의 모델은 인코더는 더 깊고, 디코더는 더 얇은 모델링이 일반적입니다.
        # 그래서 이러한 점을 없애고, -1.0 ~ +1.0 으로 유니폼하게 초기화를 해주는 방법으로 바꿨습니다.
        # => 저는 0으로도 초기화를 했는데 괜찮은 성적이 나왔습니다만, 일반적인 방법은 유니폼하게 초기화를 해주는 겁니다.
        # 디코더의 히든 스테이트의 형상이 LxBxH 이므로, 해당 형상에 맞춰서 초기화를 해줍니다.
        decoder_hidden = torch.FloatTensor(self.layer_size, batch_size, self.hidden_size).uniform_(-0.1, 0.1)#.cuda()
        # Decide Use Teacher Forcing or Not
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            # 티쳐 포싱 사용시에는, 모든 인풋 (<eos> 제외) 을 받아서 한 번에 _forward_step을 진행합니다.
            decoder_input = inputs[:, :-1]  # except </s>
            """ if teacher_forcing, Infer all at once """
            predicted_softmax = self._forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function)
            """Extract Output by Step"""
            for di in range(predicted_softmax.size(1)):
                # 한 번에 뽑은 아웃풋을 하나씩 가져오기만 합니다.
                step_output = predicted_softmax[:, di, :]
                # 가져온 놈을 decoder_results에 순서대로 넣습니다.
                decode_results.append(step_output)
        else:
            # unsqueeze() 같은 경우는 차원을 늘리는 함수입니다.
            # 보통 어떤 함수가 4차원을 처리하는데 현재 3차원을 사용할 때 차원을 4차원으로 늘리는 등에 사용됩니다.
            # 반대로 4차원의 변수를 3차원으로 압축할 때는 squeeze() 함수를 사용하면 됩니다.
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                # 티쳐 포싱 미사용시는 타입스텝마다 하나씩 뽑습니다.
                predicted_softmax = self._forward_step(decoder_input, decoder_hidden, encoder_outputs, function=function)
                # unsqueeze로 늘려준 차원을 squeeze로 다시 줄여줍니다.
                step_output = predicted_softmax.squeeze(1)
                decode_results.append(step_output)
                decoder_input = decode_results[-1].topk(1)[1]
        
        logit = torch.stack(decode_results, dim=1).to(self.device)
        y_hats = logit.max(-1)[1]
        return y_hats, logit

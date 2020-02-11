import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function = F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        """ 학습 전에 호출하면 됩니다. 학습 속도를 더 빠르게 해준다고 하네요. """
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()


    def forward(self, feats, targets=None, teacher_forcing_ratio=0.99):
        """
        인코더와 디코더의 forward()를 호출하여 결과를 반환하는 부분입니다.

        ※ 사용법 ※

        >>> enc = EncoderRNN(파라미터 처리)
        >>> dec = DecoderRNN(파라미터 처리)
        >>> seq2seq = Seq2seq(enc, dec)
        >>> seq2seq() # seq2seq 모델 포워드를 호출하는 방법이에요.
        """
        encoder_outputs, encoder_hidden = self.encoder(feats)
        y_hat, logit = self.speller(inputs = targets,
                                    encoder_hidden = encoder_hidden,
                                    encoder_outputs = encoder_outputs,
                                    function = self.decode_function,
                                    teacher_forcing_ratio = teacher_forcing_ratio)
        return y_hat, logit

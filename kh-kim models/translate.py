import argparse
import sys
import codecs
from operator import itemgetter

import torch

from data_loader import DataLoader
import data_loader
from simple_nmt.seq2seq import Seq2Seq
#from simple_nmt.transformer import Transformer
from hyperparams import TranslateHyperparams
from simple_nmt.search import SingleBeamSearchSpace


def read_text():
    print("===== 0 def read_text =====")
    # This method gets sentences from standard input and tokenize those.
    lines = []
    text = ["용기 있는 직원들을 저희가 만나서 목소리를 들어봤습니다.","자신의 의사를 정확하게 표현하는 것이 중요합니다.","저는 이 배가 진짜 항해를 했던 시기와 여러 가지를 똑같이 복원하려고 노력했어요.",
            "경찰은 방 출입문 부근에서 불이 나자 A 씨가 이를 피하려다 변을 당한 것으로 보고 있다.","화재 발생 시 대피 등 안전교육은 어떻게 하시나요?"]
    # print("===== 0.5 SYS.STDIN 이전 =====")
    #
    # sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())
    #
    # print("===== 0.5 SYS.STDIN 이후 =====")
    # print("===== SYS.STDIN print =====", sys.stdin)

    for line in text:
    #for line in sys.stdin:
        print("line:", line)
        if line.strip() != 'End':
            lines += [line.strip().split(' ')]
            print("lines : ", lines)
        else:
            break

    print("===== 1 def read_text end, RETURN LINES =====", lines)
    return lines


def to_text(indice, vocab):
    print("===== 2 def to_text(indice, vocab) =====")
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]
    print("lines : ", lines)
    print("===== 3 def to_text end, RETURN LINES =====")
    return lines


if __name__ == '__main__':
    print("===== 4  __main__ =====")
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    config = TranslateHyperparams()
    print("===== 5 config ===== : ", config)
    # Load saved model.
    saved_data = torch.load(config.model, map_location='cpu')# if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)
    print("===== 6 saved_data =====")
    # Load configuration setting in training.
    train_config = saved_data['config']
    print("===== 7 train_config ===== : ", train_config)

    if train_config.dsl:
        print("===== 8 if train_config.dsl =====")
        assert config.lang is not None

        if config.lang == train_config.lang:
            print("===== 9 if config.lang == train_config.lang TRUE =====")
            is_reverse = False
        else:
            print("===== 10 if train_config.dsl ELSE =====")
            is_reverse = True

        if not is_reverse:
            print("===== 11 if not is_reverse =====")
            # Load vocabularies from the model.
            src_vocab = saved_data['src_vocab']
            tgt_vocab = saved_data['tgt_vocab']
        else:
            print("===== 12 if not is_reverse ELSE =====")
            src_vocab = saved_data['tgt_vocab']
            tgt_vocab = saved_data['src_vocab']
    else:
        print("===== 13 train_config.dsl ELSE =====")
        # Load vocabularies from the model.
        src_vocab = saved_data['src_vocab']
        tgt_vocab = saved_data['tgt_vocab']

    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    print("===== 14 =====")
    loader = DataLoader()
    print("===== 15 =====")
    loader.load_vocab(src_vocab, tgt_vocab)
    print("===== 16 =====")
    input_size = len(loader.src.vocab)
    output_size = len(loader.tgt.vocab)

    print("===== 17 input size :  =====", input_size)
    print("===== 17 output_size :  =====", output_size)

    # Declare sequence-to-sequence model.
    # if train_config.use_transformer:
    #     model = Transformer(
    #         input_size,
    #         train_config.hidden_size,
    #         output_size,
    #         n_splits=train_config.n_splits,
    #         n_enc_blocks=train_config.n_layers,
    #         n_dec_blocks=train_config.n_layers,
    #         dropout_p=train_config.dropout,
    #     )

    model = Seq2Seq(input_size,
                    train_config.word_vec_size,
                    train_config.hidden_size,
                    output_size,
                    n_layers=train_config.n_layers,
                    dropout_p=train_config.dropout,
                    #search=SingleBeamSearchSpace()
                    )
    print("===== 18 model :  =====", model)

    if train_config.dsl:
        if not is_reverse:
            print("===== 19 if not is_reverse =====")
            model.load_state_dict(saved_data['model'][0])
        else:
            print("===== 20 if not is_reverse ELSE=====")
            model.load_state_dict(saved_data['model'][1])
    else:
        print("===== 21 train_config.dsl ELSE =====")
        model.load_state_dict(saved_data['model'])  # Load weight parameters from the trained model.

    print("===== 22 MODEL.EVAL() =====")
    model.eval()  # We need to turn-on the evaluation mode, which turns off all drop-outs.
    print("===== 23 MODEL.EVAL() END =====")

    # We don't need to draw a computation graph, because we will have only inferences.
    torch.set_grad_enabled(False)

    # Put models to device if it is necessary.
    # if config.gpu_id >= 0:
    #     model.cuda(config.gpu_id)

    # Get sentences from standard input.
    print("===== 23 LINES -- READ_TEXT() 전 =====")
    lines = read_text()
    print("===== 24 LINES[:10] =====", lines[:10])

    with torch.no_grad():  # Also, declare again to prevent to get gradients.
        print("===== 25 LINES with torch.no_grad =====")
        while len(lines) > 0:
            print("===== 26 while문 =====")
            # Since packed_sequence must be sorted by decreasing order of length,
            # sorting by length in mini-batch should be restored by original order.
            # Therefore, we need to memorize the original index of the sentence.
            sorted_lines = lines[:config.batch_size]
            lines = lines[config.batch_size:]

            print("===== 27 lines[config.batch_size:] : ", lines)

            lengths = [len(_) for _ in sorted_lines]
            orders = [i for i in range(len(sorted_lines))]

            sorted_tuples = sorted(zip(sorted_lines, lengths, orders), 
                                   key=itemgetter(1),
                                   reverse=True
                                   )
            print("===== 28 sorted_tuples : ", sorted_tuples)
            sorted_lines = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            print("===== 29 sorted_lines : ", sorted_lines)

            lengths = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            orders = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]
            print("===== 30 lengths : ", lengths)
            print("===== 30 orders : ", orders)

            # Converts string to list of index.
            x = loader.src.numericalize(loader.src.pad(sorted_lines), device='cpu')
                                        #device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
                                        #)
            print("===== 31 x : ", x)

            if config.beam_size == 1:
                print("===== 32 if config.beam_size == 1: ")
                # Take inference for non-parallel beam-search.
                y_hat, indice = model.search(x)
                print("===== 33 y_hat: ",y_hat)
                print("===== 33 indice: ", indice)

                output = to_text(indice, loader.tgt.vocab)
                print("===== 34 output: ", output)

                sorted_tuples = sorted(zip(output, orders), key=itemgetter(1))
                print("===== 35 sorted_tuples: ", sorted_tuples)

                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
                print("===== 36 output: ", output)

                sys.stdout.write('\n'.join(output) + '\n')
            else:
                print("===== 37 ELSE ===== ")
                # Take mini-batch parallelized beam search.
                batch_indice, _ = model.batch_beam_search(x,
                                                          beam_size=config.beam_size,
                                                          max_length=config.max_length,
                                                          n_best=config.n_best,
                                                          length_penalty=config.length_penalty,
                                                          )

                # Restore the original orders.
                output = []
                print("===== 38 FOR문 전 ===== ")
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                print("===== 39 FOR문 후 ===== ")

                sorted_tuples = sorted(zip(output, orders), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                print("===== 40 sorted_tuples ===== ",sorted_tuples)
                print("===== 40 output ===== ", output)

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')
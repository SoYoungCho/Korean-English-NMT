import queue
import torch.nn as nn
import torch.optim as optim
import torch
import time
#from definition import * --> 지워버리고 아래 바로 추가 함 (HYPERPARAMS에도 추가함)
#from data.split_dataset import split_dataset
#from loader.baseLoader import BaseDataLoader
#from loader.loader import load_data_list, load_targets
#from loader.multiLoader import MultiLoader
from models.DecoderRNN import DecoderRNN
from models.EncoderRNN import EncoderRNN
from models.seq2seq import Seq2seq
from train.evaluate import evaluate
from train.training import train
from hyperParams import HyperParams
from data import Dataset, DataLoader

import logging, sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")

train_set = Dataset()
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)


# 메인 함수 간소화된 버젼
if __name__ == '__main__':
    hparams = HyperParams() # 하이퍼파라미터 불러오기
    cuda = hparams.use_cuda and torch.cuda.is_available() # 쿠다 사용여부 & 쿠다 사용 가능여부
    device = torch.device('cuda' if cuda else 'cpu') # 쿠다 사용 가능시 gpu 사용 아니면 cpu 사용

    # 입력 데이터의 피쳐 사이즈
    feat_size = 33

    # EncoderRnn (인코더) 생성
    encoder = EncoderRNN(feat_size=feat_size, hidden_size=hparams.hidden_size,
                        dropout_p=hparams.dropout, layer_size=hparams.encoder_layer_size,
                        bidirectional=hparams.use_bidirectional, rnn_cell='gru') # use_pyramidal=hparams.use_pyramidal)

    # Decoder (디코더) 생성
    decoder = DecoderRNN(vocab_size=len(char2index), max_len=hparams.max_len,
                      hidden_size=hparams.hidden_size * (2 if hparams.use_bidirectional else 1),
                      sos_id=SOS_token, eos_id=EOS_token, layer_size = hparams.speller_layer_size,
                      rnn_cell = 'gru', dropout_p = hparams.dropout, use_attention = hparams.use_attention)

    # Seq2seq 모델 생성
    model = Seq2seq(encoder=encoder, decoder=decoder)

    # 파라미터 일자로 펼치기 (DataParallel에 넣어주려면 선행되어야함)
    model.flatten_parameters()

    # DataParrallel에 추가 (Multi-GPU 환경에서 사용할 경우)
    model = nn.DataParallel(model).to(device)

    # Optimize Adam Algorithm
    optimizer = optim.Adam(model.module.parameters(), lr=hparams.lr)

    # CrossEntropy로 Cost 계산
    loss_func = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    '''
    # load audio_paths & label_paths (입력 데이터 & 레이블 로딩)
    audio_paths, label_paths = load_data_list(data_list_path=TRAIN_LIST_PATH, dataset_path=DATASET_PATH)

    # {label_filenum : "2 0 5 4 2 0 5 4 ..." }
    # 레이블 파일명과 해당 파일 레이블을 Dictionary 형태로 불러오는 부분
    target_dict = load_targets(label_paths)

    # Training, Validation 데이터로 쪼개는 부분 (valid_ratio만큼 Validation 데이터 설정)
    train_batch_num, train_dataset_list, valid_dataset = \
        split_dataset(hparams, audio_paths, label_paths, valid_ratio=0.05, target_dict=target_dict)
    '''




    logger.info('start')
    train_begin = time.time()

    # 학습 시작 (max_epoch만큼 반복)
    for epoch in range(hparams.max_epochs):
        # 여기서 worker는 cpu를 의미 (멀티프로세스 환경에서만 의미 있음)
        train_queue = queue.Queue(hparams.worker_num * 2)
        # 매 에폭마다 훈련 데이터를 shuffle 해줌
        for train_dataset in train_dataset_list:
            train_dataset.shuffle()
        # Training 데이터를 읽어오는 객체 생성
        train_loader = MultiLoader(train_dataset_list, train_queue, hparams.batch_size, hparams.worker_num)
        # Training 데이터 읽어오기 시작 (run 유도)
        train_loader.start()
        # 학습 Start
        train_loss, train_cer = train(model=model, total_batch_size=train_batch_num,
                                      queue=train_queue, loss_func=loss_func,
                                      optimizer=optimizer, device=device,
                                      train_begin=train_begin, worker_num=hparams.worker_num,
                                      print_batch=10, teacher_forcing_ratio=hparams.teacher_forcing)
        # 1 에폭 학습 결과 표시
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))
        train_loader.join() # 스레드 join

        valid_queue = queue.Queue(hparams.worker_num * 2)
        # Validation 데이터를 읽어오는 객체 생성
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, hparams.batch_size, 0)
        # Validation 데이터 읽어오기 시작 (run 유도)
        valid_loader.start()

        # Validation
        valid_loss, valid_cer = evaluate(model, valid_queue, loss_func, device)
        # Validation 결과 표시
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, valid_loss, valid_cer))
        valid_loader.join()
        # 모델 weight_file 저장
        torch.save(model, "./weight_file/epoch%s" % str(epoch))
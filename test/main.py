import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    hparams = HyperParams()
    cuda = hparams.use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    enc = Encoder()
    dec = Decoder()
    model = Seq2seq(enc, dec)

    model.flatten_parameters() # 일자로 펴 준다
    model = nn.DataParallel(model).to(device)

    optimizer = optim.Adam(model.module.parameters(), lr=hparams.lr)

    # loss function의 변수로 criterion 대게 씀. reduction은 loss를 어떻게 넘길지인데, default는 mean이지만 sum이 더 빠르다고 함. 더 정확한 것은 mean
    # ignore_index는 loss 계산시 무시할 인덱스인데, PADDING 된 것들에는 loss 계산할 필요가 없다.
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    #########
    # 데이터 로딩 & 벡터화
    # (input(한글문장) & target(영어문장) 로드, 벡터로까지 만드는 과정 필요. (sentencepiece , 임베딩))
    #########

    # 데이터 리스트를 변수에 저장. 한-영 의 딕셔너리를 만든다. (문장1. 안녕 : hi)

    # train / valid 데이터로 쪼개준다. total_time_step : 한 에폭에 몇 timestep이 도는지.
    total_time_step, train_dataset, valid_dataset = split_dataset( # split_dataset은 dataset.py의 함수 이름.
        hparams = hparams,
        kor_vectors = kor_vectors,
        eng_vectors = eng_vectors,
        valid_ratio = 0.015
    )

    for epoch in range(hparams.max_ephochs):
        train_loader = DataLoader(train_dataset)
        train_loader.start() # 배치단위로 나누어서 가져 오기. thread
        train_loss, train_bleu = train(model, train_loader,criterion) #한 에폭이 돌아가는 함수 : train 함수. 불러오기
        print(train_loss, train_bleu)
        train_loader.join() #thread 사용할 경우, 돌아가는 시간이 서로 다르기 때문에 같이 끝나게 해줌.

        valid_loader = DataLoader(valid_dataset)
        valid_loader.start()
        valid_loss, valid_bleu = evaluate(model, valid_loader, criterion)
        #train과 달리 evaluate은 gradient를 주지 않는다. 즉 back-prop (X). 단순 포워드, 한 에폭당 얼마나 좋아졌는지 / 오버피팅 나는지 확인 가능
        print(valid_loss, valid_bleu)
        valid_loader.join()

        torch.save(model, "model.pt 경로")

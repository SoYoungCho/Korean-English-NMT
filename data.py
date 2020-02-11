import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self):
    # 데이터셋의 전처리를 해주는 부분
        #train_data = pd.read_csv('C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/train.csv')
        #test_data = pd.read_csv('C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/test.csv')
        sample_data = pd.read_csv('C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/sample.csv')

        self.Kor_data = sample_data['Korean']
        self.Eng_data = sample_data['English']

    def __len__(self):
    # 데이터셋의 길이. 즉, 총 샘플의 수
        return len(self.Kor_data)

    def __getitem__(self, idx):
    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수.
        kor = torch.FloatTensor(self.Kor_data[idx])
        eng = torch.FloatTensor(self.Eng_data[idx])




dataset = Dataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(dataset.Eng_data[0])
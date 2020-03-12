import logging, sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

TRAIN_KOR_PATH = "C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/Korean_datalist.csv"
VALID_KOR_PATH =  "C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/kor_sample_5.csv"
TRAIN_ENG_PATH = "C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/English_datalist.csv_datalist.csv"
VALID_ENG_PATH = "C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/eng_sample_5.csv""

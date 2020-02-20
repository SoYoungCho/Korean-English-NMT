from data_loader import DataLoader, TranslationDataset

if __name__ == '__main__':
    #loader = DataLoader(sys.argv[1],
    #                    sys.argv[2],
    #                    (sys.argv[3], sys.argv[4]),
    #                    batch_size=8
    #                    )
    loader = DataLoader('C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/', 'C:/Users/Soyoung Cho/Desktop/NMT Project/dataset/', ('kor_sample_5.csv', 'eng_sample_5.csv'),
                        shuffle=False,
                        batch_size=8
                        )

    print("len(loader.src.vocab) : ", len(loader.src.vocab))
    print("len(loader.tgt.vocab): ", len(loader.tgt.vocab))


    for batch_index, batch in enumerate(loader.train_iter):
        if batch_index > 1:
            break

        print(batch.src)
        print(batch.tgt)
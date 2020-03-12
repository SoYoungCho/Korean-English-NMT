from data_loader import DataLoader, TranslationDataset

if __name__ == '__main__':
    #loader = DataLoader(sys.argv[1],
    #                    sys.argv[2],
    #                    (sys.argv[3], sys.argv[4]),
    #                    batch_size=8
    #                    )
    #loader = DataLoader('C:/Users/Soyoung Cho/Desktop/Test/test/', 'C:/Users/Soyoung Cho/Desktop/Test/test/', ('ko.csv', 'en.csv'),
    #                   shuffle=False,
    #                    batch_size=6
    #                    )

    loader = DataLoader('C:/Users/Soyoung Cho/Desktop/Test/test/', 'C:/Users/Soyoung Cho/Desktop/Test/test/',
                        ('korsample.csv', 'engsample.csv'),
                        shuffle=False,
                        batch_size=5
                        )

    print("len(loader.src.vocab) : ", len(loader.src.vocab))
    print("len(loader.tgt.vocab): ", len(loader.tgt.vocab))


    for batch_index, batch in enumerate(loader.train_iter):
        if batch_index > 2:
            break

        print(batch.src)
        print(batch.tgt)
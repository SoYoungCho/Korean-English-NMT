import os
from torchtext import data, datasets
from tokenizer import Korean_tokenizer, English_tokenizer

PAD, BOS, EOS = 1, 2, 3


class DataLoader():

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 exts=None,
                 batch_size=5,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=False,
                 dsl=False
                 ):

        super(DataLoader, self).__init__()

        self.src = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>', #if dsl else None,
                              eos_token='<EOS>', #if dsl else None,
                              tokenize = Korean_tokenizer()
                              )

        self.tgt = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if use_bos else None,
                              eos_token='<EOS>' if use_eos else None,
                              tokenize = English_tokenizer()
                              )

        if train_fn is not None and valid_fn is not None and exts is not None:
            print("data_loader.py - if train_fn is not None and valid_fn is not None and exts is not None: 여기에 걸렸따")
            train = TranslationDataset(path=train_fn,
                                       exts=exts,
                                       fields=[('src', self.src),
                                               ('tgt', self.tgt)
                                               ],
                                       max_length=max_length
                                       )
            print("data_loader.py - train = TranslationDataset 에 걸렸당!")

            valid = TranslationDataset(path=valid_fn,
                                       exts=exts,
                                       fields=[('src', self.src),
                                               ('tgt', self.tgt)
                                               ],
                                       max_length=max_length
                                       )
            print("data_loader.py - valid = TranslationDataset 에 걸렸당!")

            self.train_iter = data.BucketIterator(train,
                                                  batch_size=batch_size,
                                                  #device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  device = 'cpu',
                                                  shuffle=False,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=False
                                                  )
            print("data_loader.py - train_iter 에 걸렸당!")

            self.valid_iter = data.BucketIterator(valid,
                                                  batch_size=batch_size,
                                                  #device='cuda:%d' % device if device >= 0 else 'cpu',
                                                  device = 'cpu',
                                                  shuffle=False,
                                                  sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                                  sort_within_batch=False
                                                  )
            print("data_loader.py - valid_iter 에 걸렸당!")

            self.src.build_vocab(train, max_size=max_vocab)
            self.tgt.build_vocab(train, max_size=max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        print("data_loader.py - def load_vocab 함수 실행")
        self.src.vocab = src_vocab
        print("self.src.vocab", self.src.vocab)
        self.tgt.vocab = tgt_vocab
        print("self.tgt.vocab", self.tgt.vocab)


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        print("data_loader.py - class TranslationDataset - def sort_key에 들어와따!")
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        print("data_loader.py - class TranslationDataset - def __init__에 들어와따!")
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            print("data_loader.py - class TranslationDataset - if not isinstance(fields[0], (tuple, list)) 들어와따!")
            fields = [('src', fields[0]), ('trg', fields[1])]

        #if not path.endswith('.'):
        #    print("data_loader.py - class TranslationDataset - if not path.endswith('.') 들어와따!")
        #    path += '.'

        # src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        src_path, trg_path = ("train_ko.csv", "train_en.csv")
        #src_path, trg_path = ("kor_sample_5.csv", "eng_sample_5.csv")

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            print("data_loader.py - class TranslationDataset - with open(src_path, encoding='utf-8') 어쩌고 저쩌고에 들어와따!")
            print("src_path : ", src_path)
            print("trg_path : ", trg_path)

            for idx, (src_line, trg_line) in enumerate(zip(src_file, trg_file)):
                # if idx%10000 == 0:
                #      print(idx, "번째 src_line : ", src_line)
                #      print(idx, "번째 trg_line : ", trg_line)

                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()),
                                                   len(trg_line.split())
                                                   ):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

                if idx>10:
                    break

            print("data_loader.py - class TranslationDataset - with open(src_path, encoding='utf-8') 의 for문이 끝났다!")

        super().__init__(examples, fields, **kwargs)


# if __name__ == '__main__':
#     print("data_loader.py - line 138 __main__에 들어와따!")
#     import sys
#     loader = DataLoader(sys.argv[1],
#                         sys.argv[2],
#                         (sys.argv[3], sys.argv[4]),
#                         batch_size=8
#                         )
#     print("data_loader.py - line 146 loader가 완료 되었따.")
#
#     print(len(loader.src.vocab))
#     print(len(loader.tgt.vocab))
#
#     for batch_index, batch in enumerate(loader.train_iter):
#         print(batch.src)
#         print(batch.tgt)
#
#         if batch_index > 1:
#             break
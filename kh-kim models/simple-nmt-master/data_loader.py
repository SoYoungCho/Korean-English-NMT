import os
import sys
import pandas as pd
from torchtext import data, datasets
from kor_tokenizer import Korean_tokenizer

PAD, BOS, EOS = 1, 2, 3

class DataLoader():

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 exts=None,
                 batch_size=64,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 dsl=False
                 ):

        super(DataLoader, self).__init__()

        self.src = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if dsl else None,
                              eos_token='<EOS>' if dsl else None,
                              tokenize = Korean_tokenizer()
                              )

        self.tgt = data.Field(sequential=True,
                              use_vocab=True,
                              batch_first=True,
                              include_lengths=True,
                              fix_length=fix_length,
                              init_token='<BOS>' if use_bos else None,
                              eos_token='<EOS>' if use_eos else None
                              )

        #if train_fn is not None and valid_fn is not None and exts is not None:
        train = TranslationDataset(path=train_fn,
                                   exts=exts,
                                   fields=[('src', self.src),
                                           ('tgt', self.tgt)
                                           ],
                                   max_length=max_length
                                   )
        valid = TranslationDataset(path=valid_fn,
                                   exts=exts,
                                   fields=[('src', self.src),
                                           ('tgt', self.tgt)
                                           ],
                                   max_length=max_length
                                   )

        self.train_iter = data.BucketIterator(train,
                                              batch_size=batch_size,
                                              #device='cuda:%d' % device if device >= 0 else 'cpu',
                                              shuffle=False,
                                              sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                              # key가 중복되지 않도록 하기 위함일듯?
                                              sort_within_batch=False
                                              )
        self.valid_iter = data.BucketIterator(valid,
                                              batch_size=batch_size,
                                              #device='cuda:%d' % device if device >= 0 else 'cpu',
                                              shuffle=False,
                                              sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                                              sort_within_batch=False
                                              )

        self.src.build_vocab(train, max_size=max_vocab)
        self.tgt.build_vocab(train, max_size=max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab


class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
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
            fields = [('src', fields[0]), ('trg', fields[1])]

        #if not path.endswith('.'):
        #    path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip() #앞 뒤 공백 제거
                #print("------------------")
                #print("src_line : ", src_line)
                #print("trg_line : ", trg_line)

                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())): #띄어쓰기 기준 분리
                    continue

                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))


        #data.Example : defines a single training/test example.
        #example의 각 열을 attribute로 한다.
        # fromlist(data, fields)

        super().__init__(examples, fields, **kwargs)




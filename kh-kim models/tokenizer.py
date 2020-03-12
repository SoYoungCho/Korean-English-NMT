import pandas as pd
import sentencepiece as spm
import MeCab


def Korean_tokenizer():
    m = MeCab.Tagger()
    delete_tag = ['BOS/EOS', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC']

    def remove_josa(sentence):
        sentence_split = sentence.split()  # 원본 문장 띄어쓰기로 분리

        dict_list = []

        for token in sentence_split:  # 띄어쓰기로 분리된 각 토큰 {'단어':'형태소 태그'} 와 같이 딕셔너리 생성
            m.parse('')
            node = m.parseToNode(token)
            word_list = []
            pos_list = []

            while node:
                morphs = node.feature.split(',')
                word_list.append(node.surface)
                pos_list.append(morphs[0])
                node = node.next
            dict_list.append(dict(zip(word_list, pos_list)))

        for dic in dict_list:  # delete_tag에 해당하는 단어 쌍 지우기 (조사에 해당하는 단어 지우기)
            for key in list(dic.keys()):
                if dic[key] in delete_tag:
                    del dic[key]

        combine_word = [''.join(list(dic.keys())) for dic in dict_list]  # 형태소로 분리된 각 단어 합치기
        result = ' '.join(combine_word)  # 띄어쓰기로 분리된 각 토큰 합치기

        return result  # 온전한 문장을 반환

    data = pd.read_csv("test.csv")
    #print(KOR_data.tail)
    KOR_data = data['Korean']

    f = open("kor_no_josa.txt", "w", encoding='utf-8')
    for row in KOR_data[:100000]:
        f.write(remove_josa(row))  # 조사 제거한 문장 저장
        f.write('\n')
    f.close()

    spm.SentencePieceTrainer.Train('--input=kor_no_josa.txt \
                               --model_prefix=korean_tok \
                               --vocab_size=100000 \
                               --hard_vocab_limit=false')

    sp = spm.SentencePieceProcessor()
    sp.Load('korean_tok.model')

    return lambda x: sp.EncodeAsPieces(x)


def English_tokenizer():
    data = pd.read_csv("test.csv")
    #print(ENG_data.tail)
    ENG_data = data['English']

    f = open("eng.txt", "w", encoding='utf-8')
    for row in ENG_data[:100000]:
        f.write(row)
        f.write('\n')
    f.close()

    spm.SentencePieceTrainer.Train('--input=eng.txt \
                               --model_prefix=english_tok \
                               --vocab_size=100000\
                               --hard_vocab_limit=false')

    sp = spm.SentencePieceProcessor()
    sp.Load('english_tok.model')

    return lambda x: sp.EncodeAsPieces(x)
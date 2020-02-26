# coding=utf-8
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import MeCab


def main():
    text = 'テレビでサッカーの試合を見る。'

    tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-whole-word-masking')
    tokenized_text = tokenizer.tokenize(text)
    print('bert  wakatigaki:{}'.format(tokenized_text))

    mecab = MeCab.Tagger("-Owakati")
    mecab_text = mecab.parse(text)
    print('mecab wakatigaki:{}'.format(mecab_text.split()))


if __name__ == '__main__':
    main()

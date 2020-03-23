import csv
import jieba
import torch

from collections import Counter
from args import args

STOPWORDS = []
with open(args.stop_words, 'r') as f:
    for i in f:
        STOPWORDS.append(i[:-1])

def read_file(source_file):
    negative, neutral, positive = [], [], []
    with open(source_file, encoding='gb18030', mode='r', errors='ignore') as f:
        reader = csv.reader(f)

        for line in reader:
            try:
                idx, sent, label = int(line[0]), line[3], int(line[6]) + 1
            except:
                continue

            if label == 0:
                negative.append(sent)
            elif label == 1:
                neutral.append(sent)
            elif label == 2:
                positive.append(sent)
            else:
                continue
        
    return (negative, neutral, positive)

def build_semantic_vocab(negative, neutral, positive):
    result = [[], [], []] # contains the tokens among all the three corpus

    for i, corpus in enumerate((negative, neutral, positive)):
        for sent in corpus:
            result[i] += [word for word in list(jieba.cut(sent)) if word not in STOPWORDS]
        
    negative, neutral, positive = map(lambda x: Counter(x), result)
    torch.save((negative, neutral, positive), 'tmp.pt')

    ## to get the top n: n+k tokens, just type:
    ## negative.most_common()[n: n+k]
    
    return (negative, neutral, positive)

if __name__ == "__main__":
    out = read_file(args.raw_train_data)
    result = build_semantic_vocab(*out)
    # result = torch.load('tmp.pt')


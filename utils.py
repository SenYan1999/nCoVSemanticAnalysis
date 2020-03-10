import torch
import pickle
import logging
import csv
import jieba

from tqdm import tqdm
from logging import handlers
from torch.utils.data import Dataset
from transformers import BertTokenizer
from args import args

def save_pt(source, target):
    with open(target, 'wb') as f:
        pickle.dump(source, f)

def load_pt(file):
    with open(file, 'rb') as f:
        result = pickle.load(f)
    return result

def init_logger(filename, when='D', backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
    logger = logging.getLogger(filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(th)

    return logger

logger = init_logger(filename=args.log_file)

class BaseDataset(Dataset):
    def __init__(self, source, max_len, bert_type):
        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        data = self._convert_source_words(source)
        self.idxs, self.sents, self.labels = self.convert_data(data, max_len)

    def _convert_source_words(self, source):
        data = []
        count = 0

        with open(source, 'r', encoding='gb18030', errors='ignore') as f:
            reader = csv.reader(f)
            next(reader) # skip the first head line
            for line in reader:
                try:
                    idx, sent, label = int(line[0]), line[3], int(line[6]) + 1
                except:
                    count += 1
                    continue
                if label not in [0, 1, 2]:
                    count += 1
                    continue

                data.append((idx, sent, label))
        
        logger.info('Preprocess: %d | Ignore: %d' % (len(data), count))

        return data

    def convert_data(self, data, max_len):
        idxs, sents, labels = [], [], []
        for line in tqdm(data, total=len(data)):
            idx, sent, label = line[0], line[1], line[2]

            sent = self.tokenizer.tokenize(sent)
            sent = ['[CLS]'] + sent + ['[SEP]']
            if len(sent) < max_len:
                sent += ['[PAD]' for _ in range(max_len - len(sent))]
            else:
                sent = sent[:max_len]
            sent = self.tokenizer.convert_tokens_to_ids(sent)
            
            idxs.append(idx)
            sents.append(sent)
            labels.append(label)

        idxs, sents, labels = torch.LongTensor(idxs), torch.LongTensor(sents), \
            torch.LongTensor(labels)

        return idxs, sents, labels
    
    def __getitem__(self, index):
        return (self.idxs[index], self.sents[index], self.labels[index])

    def __len__(self):
        return self.sents.shape[0]

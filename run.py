import torch

from args import args
from utils import BaseDataset, logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import BaseBertModel
from trainer import Trainer

def preprocess():
    dataset = BaseDataset(args.raw_train_data, args.max_len, args.bert_type)
    torch.save(dataset, args.train_data)

def train():
    # prepare dataset and dataloader
    dataset = torch.load(args.train_data)
    train_size = int(len(dataset) * (1 - args.eval_rate))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
        DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    
    # define model and optimzier
    model = BaseBertModel(args.bert_type, args.d_bert, args.num_class)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # define trainer and begin training
    trainer = Trainer(train_dataloader, eval_dataloader, model, optimizer)
    trainer.train(args.num_epoch, args.save_path)

if __name__ == "__main__":
    if args.preprocess:
        preprocess()
    elif args.train:
        train()
    else:
        print('Please choose a running mode.')

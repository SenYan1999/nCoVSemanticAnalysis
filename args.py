import argparse

parser = argparse.ArgumentParser()

# run mode
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--train', action='store_true')

# model type
parser.add_argument('--bert', action='store_true')

# data preprocess
parser.add_argument('--raw_train_data', type=str, default='data/nCoV_100k_train.labled.csv')
parser.add_argument('--train_data', type=str, default='data/train.pt')
parser.add_argument('--max_len', type=int, default='100')
parser.add_argument('--min_occurance', type=int, default='1')

# model
parser.add_argument('--bert_type', type=str, default='bert-base-chinese')
parser.add_argument('--d_bert', type=int, default=768)
parser.add_argument('--num_class', type=int, default=3)
parser.add_argument('--drop_out', type=float, default=0.2)

# train
parser.add_argument('--eval_rate', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-5)

# save & log
parser.add_argument('--log_file', type=str, default='log/log.log')
parser.add_argument('--save_path', type=str, default='save_model/')

# parse args
args = parser.parse_args()
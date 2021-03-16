import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
from tensorboardX import SummaryWriter
import dataloader
import model


#use gpu whenever available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#loading model checkpoints
def load_ckp(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model



def main():
    #building vocab
    train_df = pd.read_csv("data/train.tsv", sep ='\t')
    train_df['boilerplate'].replace(to_replace=r'"title":', value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'"body":', value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'"url":',value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'{|}|"',value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'\\u00a0',value="",inplace=True,regex=True)
    train_df['boilerplate']=train_df['boilerplate'].str.lower()
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train_df['boilerplate']:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1, vectors='glove.6B.300d')


    #prepare the raw data
    test_df = pd.read_csv("data/test.tsv", sep ='\t')
    test_df['boilerplate'].replace(to_replace=r'"title":', value="",inplace=True,regex=True)
    test_df['boilerplate'].replace(to_replace=r'"body":', value="",inplace=True,regex=True)
    test_df['boilerplate'].replace(to_replace=r'"url":',value="",inplace=True,regex=True)
    test_df['boilerplate'].replace(to_replace=r'{|}|"',value="",inplace=True,regex=True)
    test_df['boilerplate'].replace(to_replace=r'\\u00a0',value="",inplace=True,regex=True)
    test_df['boilerplate']=test_df['boilerplate'].str.lower()

    #create and load the pretrained refinenet model
    net = model.tncnet(vocab).to(device)
    last_model_path = "model/epoch_last.pt"
    net = load_ckp(last_model_path, net)
    net.eval()

    #data loader
    data_loader = dataloader.dataLoader(test_df, vocab, batch_size=len(test_df), isTrain=False)

    for idx, (text_index, offsets, numeric, label) in enumerate(data_loader):
        text_index = text_index.to(device)
        offsets = offsets.to(device)
        numeric = numeric.to(device)
        outputs = net(text_index, offsets, numeric)
        pred_label = torch.round(outputs)
        pred_label = np.asarray(pred_label.squeeze().detach().to('cpu'))
        test_df['label']=pred_label
        test_df.to_csv('Indranil_output_final.csv',columns=['urlid','label'],index=False)
    return


if __name__ == "__main__":
    main()

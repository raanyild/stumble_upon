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


#saving and loading model checkpoints
def save_ckp(state, path):
    torch.save(state, path)
    return 

def load_ckp(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def train(net, data_loader, criterion, optimizer, start_epoch=0, end_epoch=0, model_path=None, log_path=None):
    experiment_name = 'tncnet'
    writer = SummaryWriter(log_path + experiment_name)
    iteration = 0
    for epoch in range(start_epoch, end_epoch):
        total_loss = 0.0
        for i, (text_index, offsets, numeric, label) in enumerate(data_loader):
            text_index = text_index.to(device)
            offsets = offsets.to(device)
            numeric = numeric.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = net(text_index, offsets, numeric)
            loss = criterion(outputs.squeeze(), label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print("Epoch {0} & Iteration {1} : current loss {2}".format(epoch, iteration, loss.item()))
            iteration += 1

        total_loss /= len(data_loader)
        writer.add_scalar('training_loss', total_loss, epoch)

        if(epoch%5 == 4):
            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
            ckp_path = model_path + "epoch_" + str(epoch) + ".pt"
            save_ckp(checkpoint, ckp_path)
            
    return



def main():
    #prepare the raw data
    train_df = pd.read_csv("data/train.tsv", sep ='\t')
    train_df['boilerplate'].replace(to_replace=r'"title":', value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'"body":', value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'"url":',value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'{|}|"',value="",inplace=True,regex=True)
    train_df['boilerplate'].replace(to_replace=r'\\u00a0',value="",inplace=True,regex=True)
    train_df['boilerplate']=train_df['boilerplate'].str.lower()

    #building the vocab
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for line in train_df['boilerplate']:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1, vectors='glove.6B.300d')


    #data loader
    data_loader = dataloader.dataLoader(train_df, vocab, batch_size=32, isTrain=True)

    #create the NN model
    net = model.tncnet(vocab).to(device)

    # train the network epoch 1-200
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.4, 0.999), eps=1e-08)
    criterion = nn.BCELoss().to(device)
    train(net=net, data_loader=data_loader, criterion=criterion, optimizer=optimizer, start_epoch=0, end_epoch=200,
        model_path="model/", log_path="log_dir/")
    return

if __name__ == "__main__":
    main()
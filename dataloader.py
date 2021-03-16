import torch

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

class NumericTextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, vocab, isTrain=False):
        self.dataframe = dataframe
        self.vocab = vocab
        self.texts = dataframe['boilerplate']
        self.numerics = dataframe[['avglinksize', 'commonlinkratio_1',
                                   'commonlinkratio_2', 'commonlinkratio_3', 'commonlinkratio_4',
                                   'compression_ratio', 'embed_ratio', 'frameTagRatio',
                                   'hasDomainLink', 'html_ratio', 'image_ratio',
                                   'lengthyLinkDomain', 'linkwordscore',
                                   'non_markup_alphanum_characters', 'numberOfLinks', 'numwords_in_url',
                                   'parametrizedLinkRatio', 'spelling_errors_ratio']]
        self.isTrain = isTrain
        self.labels = None
        if isTrain:
            self.labels = dataframe['label']
        
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, index):
        text = self.texts[index]
        text_list = tokenizer(text)
        text_index = [self.vocab[token] for token in text_list]
        numeric = list(self.numerics.iloc[index])
        label = -1
        if self.isTrain:
            label = self.labels[index]
        return text_index, numeric, label


def collate_batch(batch):
    text_index_, numeric_, labels_, offsets = [], [], [], [0]
    for (t_, n_, l_) in batch:
        text_index_.append(torch.tensor(t_))
        offsets.append(len(t_))
        numeric_.append(n_)
        labels_.append(l_)
    text_index_t = torch.cat(text_index_)
    offsets_t = torch.tensor(offsets[:-1]).cumsum(dim=0)
    labels_t = torch.tensor(labels_, dtype=torch.int64)
    numeric_t = torch.tensor(numeric_)
    return text_index_t, offsets_t, numeric_t, labels_t


def dataLoader(train_df, vocab, batch_size=32, isTrain=False, shuffle=False):
    params = {'batch_size': batch_size,
          'shuffle': shuffle,
          'num_workers': 8}
    tndataset = NumericTextDataset(train_df, vocab, isTrain)
    data_loader = torch.utils.data.DataLoader(tndataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return data_loader
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os, re
import nltk
from collections import Counter
from build_vocab import Vocabulary, build_vocab

def create_captions(filepath):

    # clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{},0-9]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()

    captions = {}

    with open(filepath, "r") as file:

        for line in file:
            line = line.replace("\n", "").split("\t")
            file_name = re.sub('\D', '', line[0])
            sentence_tokens = []
            for sentence in line[1].split("."):
                tokens = bioclean(sentence)
                if len(tokens) == 0:
                    continue
                caption = " ".join(tokens)
                sentence_tokens.append(caption)
            
            captions[file_name] = (sentence_tokens)
    
    return captions

class iuxray(Dataset):
    def __init__(self, root_dir, tsv_path, image_path, fea_path, vocab = None, transform=None):
        self.root_dir = root_dir
        self.tsv_path = tsv_path
        self.image_path = image_path
        self.fea_dir = fea_path
        tsv_file = os.path.join(self.root_dir, self.tsv_path)
        
        self.captions = create_captions(tsv_file)
        if vocab is None:
            self.vocab = build_vocab(self.captions, 1)
        else:
            self.vocab = vocab
        self.data_file = pd.read_csv(tsv_file, delimiter='\t',encoding='utf-8')
        self.transform = transform

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, idx):
        # print(idx)
        pretrain = True
        if pretrain:
            fea_name = re.sub('\D', '', self.data_file.iloc[idx, 0]) + '.npy'
            fea_path = os.path.join(self.fea_dir, fea_name)
            fea_fc = np.load(fea_path)
            fea_fc = torch.from_numpy(fea_fc)
            img_name = os.path.join(self.root_dir, self.image_path, self.data_file.iloc[idx, 0])
        else:
            img_name = os.path.join(self.root_dir, self.image_path, self.data_file.iloc[idx, 0])
            image = Image.open(img_name)
            if self.transform is not None:
                image = self.transform(image)
        
        caption = self.captions[fea_name.split('.')[0]]
        captions = ". ".join(caption)

        tokens = nltk.tokenize.word_tokenize(str(captions).lower())
        sentence = []
        if pretrain:
            sentence.append(0)
            sentence.extend([self.vocab(token) for token in tokens])
        else:
            sentence.append(self.vocab('<start>'))
            sentence.extend([self.vocab(token) for token in tokens])
            sentence.append(self.vocab('<end>'))
        # print([self.vocab.idx2word[k] for k in sentence])
        # print([self.vocab.idx2word[sentences[0][k]] for k in sentences[0]])
        # for i in range(len(sentences)):
        #     if len(sentences[i]) < max_sent_len:
        #         sentences[i] = sentences[i] + (max_sent_len - len(sentences[i]))* [self.vocab('<pad>')]
                
        target = torch.Tensor(sentence)

        if pretrain:
            return fea_fc, target, img_name
        else:
            return image, target, img_name


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption, no_of_sent, max_sent_len).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption, no_of_sent, max_sent_len). 
            - image: torch tensor of shape (3, crop_size, crop_size).
            - caption: torch tensor of shape (no_of_sent, max_sent_len); variable length.
            - no_of_sent: number of sentences in the caption
            - max_sent_len: maximum length of a sentence in the caption

    Returns:
        images: torch tensor of shape (batch_size, 3, crop_size, crop_size).
        targets: torch tensor of shape (batch_size, max_no_of_sent, padded_max_sent_len).
        prob: torch tensor of shape (batch_size, max_no_of_sent)
    """
    # Sort a data list by caption length (descending order).
#     data.sort(key=lambda x: len(x[1]), reverse=True)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, img_name = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    try:
        images = torch.stack(images, 0)
    except RuntimeError as e:
        print(e)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), 52).long()
    masks = torch.zeros(len(captions), 52)
    for i, cap in enumerate(captions):
        end = 51 if lengths[i] > 51 else lengths[i]
        targets[i, :end] = cap[:end]
        masks[i, :end+1] = 1

    return images, targets, lengths, masks, img_name

def get_loader(root_dir, tsv_path, image_path, fea_path, transform, batch_size, shuffle, num_workers, vocab = None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # dataset
    data = iuxray(root_dir = root_dir, 
             tsv_path = tsv_path, 
             image_path = image_path,
             fea_path = fea_path,
             vocab = vocab,
             transform = transform)
    
    # Data loader for dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, resize_length, resize_width).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset = data, 
                                              batch_size = batch_size,
                                              shuffle = shuffle,
                                              num_workers = num_workers,
                                              collate_fn = collate_fn)

    return data_loader, data.vocab
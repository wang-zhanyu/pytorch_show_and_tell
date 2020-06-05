# -*- coding = utf-8 -*-
# @Time: 2020/5/6 21:06
# @Author: zhanyu Wang
# @File: prepro_labels_peir.py
# @Software: PyCharm

"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image,
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
import skimage.io
from PIL import Image
import re
import nltk
from build_vocab import Vocabulary, build_vocab


def create_captions(filepath):
    ## the captions have the impression and findings concatenated to form one big caption
    ## i.e. caption = impression + " " + findings
    ## WARNING: in addition to the XXXX in the captions, there are <unk> tokens

    # clean for BioASQ
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{},0-9]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()
    captions = []
    with open(filepath, "r") as file:
        for line in file:
            line = line.replace("\n", "").split("\t")

            sentence_tokens = []

            for sentence in line[1].split("."):
                tokens = bioclean(sentence)
                if len(tokens) == 0:
                    continue
                caption = " ".join(tokens)
                sentence_tokens.append(caption)

            captions.append(sentence_tokens)
    return captions


def main(params):

    captions = create_captions(params['tsv_file'])
    vocab = build_vocab(captions, 1)

    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{},0-9]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()
    images = []
    peirtalk = {}
    sentences = []
    label_length = []
    for filepath in params["input_path"]:
        split = filepath.split('/')[-1].split('_')[0]

        with open(filepath, "r") as file:
            for index, line in enumerate(file):
                image = {}
                line = line.replace("\n", "").split("\t")
                peir_id = re.sub('\D', '', line[0])
                image['id'] = peir_id
                image['filename'] = line[0]
                if index > 6000 and split == "train":
                    image['split'] = 'val'
                else:
                    image['split'] = split

                sentence_tokens = []
                for sentence in line[1].split("."):
                    tokens = bioclean(sentence)
                    if len(tokens) == 0:
                        continue
                    caption = " ".join(tokens)
                    sentence_tokens.append(caption)

                captions = ". ".join(sentence_tokens)
                tokens = nltk.tokenize.word_tokenize(str(captions).lower())

                image['caption'] = captions
                images.append(image)

                sentence = []
                # sentence.append(vocab('<start>'))
                sentence.extend([vocab(token) for token in tokens])
                # sentence.append(vocab('<end>'))
                length = len(sentence)
                label_length.append(length)

                sentence = np.array(sentence)
                sent = np.zeros(params['max_length'], dtype='uint32')
                if sentence.shape[0] <= params['max_length']:
                    sent[:sentence.shape[0]] = sentence
                else:
                    sent[:] = sentence[:sent.shape[0]]
                    sent[-1] = sentence[-1]

                sentences.append(sent)

    peirtalk['ix_to_word'] = vocab.idx2word
    peirtalk['images'] = images

    json.dump(peirtalk, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])

    seed(123)  # make reproducible

    L = np.array(sentences)
    label_length = np.array(label_length)
    label_start_ix = np.array([i+1 for i in range(len(images))])
    label_end_ix = np.array([i+1 for i in range(len(images))])

    # create output h5 file
    f_lb = h5py.File(params['output_h5'] + '_label.h5', "w")
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--tsv_file', default='../data/peir_gross.tsv',
                        help='all info of IU or peir')
    parser.add_argument('--input_path', default=['../data/test_images_peir.tsv', '../data/train_images_peir.tsv'],
                        help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='../data/peirtalk.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/peirtalk', help='output h5 file')
    parser.add_argument('--images_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=30, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=3, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)

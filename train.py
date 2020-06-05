import os
import torch
import argparse
from dataloader import get_loader
from torchvision import transforms
from torch import nn
import time
import numpy as np
from torchsummary import summary
from torch.optim import lr_scheduler
import logging
from torch.utils.tensorboard import SummaryWriter
from logger import Logger
from torch.nn.utils.rnn import pack_padded_sequence
from model import ShowTellModel
import sys
import json
import re
import torch.nn.functional as F
from utils import *
from metric import *


def main(args):
    Not_improve_counter = 0
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    train_loader, vocab = get_loader(args.root_dir,
                                     args.train_tsv_path,
                                     args.image_path,
                                     args.fea_path,
                                     transform,
                                     args.batch_size,
                                     args.shuffle,
                                     args.num_workers)

    vocab_size = len(vocab)
    print("vocab_size: ", vocab_size)

    val_loader, _ = get_loader(args.root_dir,
                               args.val_tsv_path,
                               args.image_path,
                               args.fea_path,
                               transform,
                               args.batch_size,
                               args.shuffle,
                               args.num_workers,
                               vocab)

    decoderRNN = ShowTellModel(vocab_size, args).to(args.device)

    if args.load_checkpoint_path is not None:
        model_state_dict = load_checkpoint(args)
        start_epoch = model_state_dict["epoch"]
        decoderRNN.load_state_dict(model_state_dict["decoderRNN"])
        print("decoderRNN loaded")
        init_loss = float(str(args.load_checkpoint_path).split("_")[-1][:-8])
    else:
        start_epoch = 0
        init_loss = 10000

    # Loss and optimizer
    criterion = LanguageModelCriterion().to(args.device)
    params_lstm = list(decoderRNN.parameters())
    optim_lstm = torch.optim.Adam(params=params_lstm, lr=args.learning_rate_lstm)
    scheduler_lstm = lr_scheduler.ReduceLROnPlateau(optim_lstm, factor=0.2, mode="min", patience=10, cooldown=10,
                                                    min_lr=1e-7, verbose=True)
    # Train the models
    total_step = len(train_loader)
    current_time = time.time()
    save_path = os.path.join(args.model_dir,
                             str(current_time).split('.')[0] + "_" + args.image_path.split("_")[0] + "_" + args.mode)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = init_logger(save_path)

    for epoch in range(start_epoch, args.num_epochs):
        decoderRNN.train()

        for i, (fea_fc, captions, lengths, masks, _) in enumerate(train_loader):
            optim_lstm.zero_grad()

            captions = captions.unsqueeze(1).to(args.device)
            masks = masks.unsqueeze(1).to(args.device)

            vis_enc_output = fea_fc.to(args.device)
            outputs = decoderRNN(vis_enc_output, captions)
            loss = criterion(outputs, captions[..., 1:], masks[..., 1:])
            loss.backward()
            optim_lstm.step()

            if args.grad_clip_value != 0:
                getattr(torch.nn.utils, 'clip_grad_%s_' %(args.grad_clip_mode))(decoderRNN.parameters(), args.grad_clip_value)
            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))

            if i>5:
                break

        evaluate(args, val_loader, decoderRNN, vocab)

        log(logger, loss, optim_lstm.param_groups[0]['lr'], epoch)

        if loss.item() < init_loss:
            torch.save({'decoderRNN': decoderRNN.state_dict(),
                        'optim_lstm': optim_lstm.state_dict(),
                        'epoch': epoch},
                       os.path.join(save_path, "model_best.pth.tar"))
            print("train_loss decrease from {} to {}".format(init_loss, loss.item()))
            print("saved model train_epoch_{}_loss_{:.4f}.pth.tar".format(epoch + 1, loss.item()))
            init_loss = loss.item()
            Not_improve_counter = 0
        else:
            Not_improve_counter += 1
            print("val loss is not decreased {} times".format(Not_improve_counter))

        scheduler_lstm.step(loss)


def evaluate(args, val_loader, decoderRNN, vocab):
    decoderRNN.eval()
    references = {}
    hypotheses = {}

    # hypothese = []
    # data = json.load(open(args.dataset, 'r'))
    # images = data['images']
    for i, (fea_fc, captions, lengths, _, img_name) in enumerate(val_loader):

        captions = captions.to(args.device)
        vis_enc_output = fea_fc.to(args.device)
        pred_words, seq_logprobs = decoderRNN.sample(vis_enc_output)

        entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((pred_words > 0).float().sum(1) + 1)
        perplexity = - seq_logprobs.gather(2, pred_words.unsqueeze(2)).squeeze(2).sum(1) / ((pred_words > 0).float().sum(1) + 1)

        for j in range(pred_words.shape[0]):
            # img_id = re.sub('\D', '', img_name[j][0].split('\\')[-1])
            img_id = re.sub('\D', '', img_name[j])
            pred_sent = pred_words[j, :].cpu().numpy()
            real_sent = captions[j, :].cpu().numpy()
            pred_sentences = " ".join([vocab.idx2word[w] for w in pred_sent if w != 0]).replace('<unk>', '.')
            real_sentences = " ".join([vocab.idx2word[w] for w in real_sent if w != 0]).replace('<unk>', '.')
            # real_sentences_2 = [im['caption'] for im in images if im['id'] == img_id]
            entry = {'image_id': img_id, 'caption': pred_sentences, 'gts':real_sentences, 'perplexity': perplexity[j].item(),
                     'entropy': entropy[j].item()}
            # hypothese.append(entry)
            hypotheses[img_id] = [pred_sentences]
            references[img_id] = [real_sentences]


        print('pred:{}\n'.format(pred_sentences))
    # language_eval(args.dataset, hypothese, args.mode, split='test')
    bleu1, bleu2, bleu3, bleu4 = bleu(references, hypotheses)
    cider_score = cider(references, hypotheses)
    rouge_score = rouge(references, hypotheses)
    print(
        '\n * BLEU-1 - {:.4f}, BLEU-2 - {:.4f}, BLEU-3 - {:.4f}, BLEU-4 - {:.4f}, CIDEr - {:.4f}, Rouge - {:.4f}\n'.format(
            bleu1,
            bleu2,
            bleu3,
            bleu4,
            cider_score,
            rouge_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset_name = 'iu_xray'
    parser.add_argument('--fea_path', type=str, default=r'E:\pHD\code\1.6- pytorch (self-critical)\data\iuxraytalk_fc', help='path for feature dir')
    parser.add_argument('--dataset', type=str, default=r'.\data\iuxraytalk.json', help='path for json file')
    parser.add_argument('--root_dir', type=str, default=r'E:\pHD\data\{}'.format(dataset_name), help='path for root dir')
    parser.add_argument('--tsv_path', type=str, default='{}.tsv'.format(dataset_name), help='path of the training tsv file')
    parser.add_argument('--train_tsv_path', type=str, default='train_images.tsv', help='path of the training tsv file')
    parser.add_argument('--val_tsv_path', type=str, default='test_images.tsv', help='path of the validation tsv file')
    parser.add_argument('--image_path', type=str, default='{}_images'.format(dataset_name), help='path of the images file')
    parser.add_argument('--img_size', type=int, default=224, help='size to which image is to be resized')
    parser.add_argument('--crop_size', type=int, default=224, help='size to which the image is to be cropped')
    parser.add_argument('--device_number', type=str, default="0", help='which GPU to run experiment on')
    parser.add_argument('--model_dir', type=str, default='./results', help='path of save trained model')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='path of save trained model')
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                        help='path of save trained model')
    parser.add_argument('--mode', type=str, default="show_tell", help='choose ["show and tell", "show attention and tell"]')
    parser.add_argument('--pair', type=bool, default=False,
                        help='whether train the model with the paired data')

    # Model settings
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=1024,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')

    parser.add_argument('--grad_clip_mode', type=str, default='value',
                    help='value or norm')
    parser.add_argument('--grad_clip_value', type=float, default=0.1,
                    help='clip gradients at this value/max_norm, 0 means no clipping')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--seq_length', type=int, default=30,
                        help='max length of the generated caption')

    parser.add_argument('--batch_size', type=int, default=16, help='size of the batch')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle the instances in dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--num_epochs', type=int, default=70, help='number of epochs to train the model')
    parser.add_argument('--learning_rate_lstm', type=int, default=4e-4, help='learning rate for LSTM Decoder')

    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args.device)

    main(args)
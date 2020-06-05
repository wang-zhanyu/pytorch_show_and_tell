from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import numpy as np
from torchvision import transforms
from dataloader import get_loader
import argparse


class ShowTellModel(nn.Module):
    def __init__(self, vocab_size, args):
        super(ShowTellModel, self).__init__()
        self.vocab_size = vocab_size
        self.input_encoding_size = args.input_encoding_size
        self.rnn_type = args.rnn_type
        self.rnn_size = args.rnn_size
        self.num_layers = args.num_layers
        self.drop_prob_lm = args.drop_prob_lm
        self.seq_length = args.seq_length
        self.fc_feat_size = args.fc_feat_size

        self.ss_prob = 0.0  # Schedule sampling probability

        self.img_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.core = getattr(nn, self.rnn_type.upper())(self.input_encoding_size, self.rnn_size, self.num_layers,
                                                       bias=False, dropout=self.drop_prob_lm)
        self.embed = nn.Embedding(self.vocab_size, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size+1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.num_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.num_layers, bsz, self.rnn_size)

    def forward(self, fc_feats, seq):
        batch_size = fc_feats.size(0)
        seq = seq.view(-1, seq.shape[-1])
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size * seq_per_img)
        outputs = []

        for i in range(seq.size(1)):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:  # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size * seq_per_img).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i - 1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i - 1].data.clone()
                        prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                        it.long().index_copy_(0, sample_ind,
                                              torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                else:
                    it = seq[:, i - 1].clone()
                    # break if all the sequences end
                if i >= 2 and seq[:, i - 1].data.sum() == 0:
                    break
                xt = self.embed(it.long())

            output, state = self.core(xt.unsqueeze(0), state)
            output = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1).contiguous()

    def get_logprobs_state(self, it, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt.unsqueeze(0), state)
        logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    xt = self.img_embed(fc_feats[k:k + 1]).expand(beam_size, self.input_encoding_size)
                elif t == 1:  # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(it)

                output, state = self.core(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, opt={}):
        sample_method = opt.get('sample_method', 'greedy')
        temperature = opt.get('temperature', 1.0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros(batch_size, self.seq_length, dtype=torch.long)
        # seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1: # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                xt = self.embed(it)

            output, state = self.core(xt.unsqueeze(0), state)
            logprobs = F.log_softmax(self.logit(self.dropout(output.squeeze(0))), dim=1)

            # sample the next word
            if t == self.seq_length + 1: # skip if we achieve maximum length
                break
            if sample_method == 'greedy':
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it #seq[t] the input of t+2 time step
                # seqLogprobs[:,t-1] = sampleLogprobs.view(-1)
                seqLogprobs[:, t - 1] = logprobs
                if unfinished.sum() == 0:
                    break

        return seq, seqLogprobs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--fea_path', type=str, default=r'E:\pHD\code\1.6- pytorch (self-critical)\data\iuxraytalk_fc',
                        help='path for feature dir')
    parser.add_argument('--root_dir', type=str, default=r'E:\pHD\data\iu_xray', help='path for root dir')
    parser.add_argument('--tsv_path', type=str, default='iu_xray.tsv', help='path of the training tsv file')
    parser.add_argument('--train_tsv_path', type=str, default='train_images.tsv', help='path of the training tsv file')
    parser.add_argument('--val_tsv_path', type=str, default='test_images.tsv', help='path of the validation tsv file')
    parser.add_argument('--image_path', type=str, default='iu_xray_images', help='path of the images file')
    parser.add_argument('--img_size', type=int, default=224, help='size to which image is to be resized')
    parser.add_argument('--crop_size', type=int, default=224, help='size to which the image is to be cropped')
    parser.add_argument('--device_number', type=str, default="0", help='which GPU to run experiment on')
    parser.add_argument('--model_dir', type=str, default='./results', help='path of save trained model')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='path of save trained model')
    parser.add_argument('--load_checkpoint_path', type=str, default=None,
                        help='path of save trained model')
    parser.add_argument('--mode', type=str, default="show and tell",
                        help='choose ["show and tell", "show attention and tell"]')

    # Model settings
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
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
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    train_loader, vocab = get_loader(args.root_dir,
                               args.train_tsv_path,
                               args.image_path,
                               transform,
                               args.batch_size,
                               args.shuffle,
                               args.num_workers)

    vocab_size = len(vocab)
    print("vocab_size: ", vocab_size)

    criterion = nn.CrossEntropyLoss().to(args.device)
    show_tell = ShowTellModel(vocab_size).to(args.device)
    params_lstm = list(show_tell.parameters())
    optim_lstm = torch.optim.Adam(params=params_lstm, lr=args.learning_rate_lstm)
    total_step = len(train_loader)
    for epoch in range(args.num_epochs):

        for i, (fea_fc, captions, lengths, _) in enumerate(train_loader):
            optim_lstm.zero_grad()

            captions = captions.to(args.device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            vis_enc_output = fea_fc.to(args.device)
            outputs = show_tell(vis_enc_output, captions)
            loss = criterion(outputs.contiguous().view(-1, vocab_size), captions.contiguous().view(-1))
            loss.backward()
            optim_lstm.step()

            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item()))
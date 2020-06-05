# -*- coding = utf-8 -*-
# @Time: 2020/6/4 17:42
# @Author: zhanyu Wang
# @File: utils.py
# @Software: PyCharm

import os
import torch
from torch import nn
from logger import Logger
import json
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
from logger import Logger

try:
    sys.path.append("cococaption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: coco-caption not available')


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        output = -input.gather(2, target.unsqueeze(2).long()).squeeze(2) * mask
        # Average over each token
        output = torch.sum(output) / torch.sum(mask)
        return output


def load_checkpoint(args):
    try:
        model_state_dict = torch.load(os.path.join(args.checkpoint_dir, args.load_checkpoint_path))
        print("[Load Model-{} Succeed!]".format(args.load_checkpoint_path))
        print("Load From Epoch {}".format(model_state_dict['epoch']))
        return model_state_dict
    except Exception as err:
        print("[Load Model Failed] {}".format(err))
        raise err


def init_logger(save_path):
    logger = Logger(os.path.join(save_path, 'logs'))
    return logger


def log(logger,
        train_loss,
        lr_lstm,
        epoch):
    info = {
        'train loss': train_loss,
        'learning rate LSTM': lr_lstm
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)


def language_eval(dataset, preds, model_id, split):
    # create output dictionary
    out = {}
    cache_path = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '.json')
    coco = COCO(dataset)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, dataset)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval

    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out
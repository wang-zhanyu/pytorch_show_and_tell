# -*- coding = utf-8 -*-
# @Time: 2020/4/8 13:46
# @Author: zhanyu Wang
# @File: metric.py
# @Software: PyCharm

from evalcap.bleu.bleu import Bleu
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from evalcap.rouge.rouge import Rouge
from evalcap.spice.spice import Spice

def bleu(gts, res):
    scorer = Bleu(n=4)
    # scorer += (hypo[0], ref1)   # hypo[0] = 'word1 word2 word3 ...'
    #                                 # ref = ['word1 word2 word3 ...', 'word1 word2 word3 ...']
    score, scores = scorer.compute_score(gts, res)
    # print('belu = %s' % score)
    return score


def cider(gts, res):
    scorer = Cider()
    # scorer += (hypo[0], ref1)
    (score, scores) = scorer.compute_score(gts, res)
    # print('cider = %s' % score)
    return score


def meteor(gts, res):
    scorer = Meteor()
    score, scores = scorer.compute_score(gts, res)
    # print('meter = %s' % score)
    return score


def rouge(gts, res):
    scorer = Rouge()
    score, scores = scorer.compute_score(gts, res)
    # print('rouge = %s' % score)
    return score


def spice(gts, res):
    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    # print('spice = %s' % score)
    return score
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn as nn


@register_criterion('classification_multi_amsoftmax')
class AmsoftmaxMultiClassifyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.s = 30
        self.m = 0.2
        self.ce = nn.CrossEntropyLoss()


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        costh_emo = net_output["costh_emo"]
        targets_emo = sample['target_emo'].view(-1).long() - 4

        label_emo = targets_emo
        label_emo_view = label_emo.view(-1, 1)
        if label_emo_view.is_cuda: label_emo_view = label_emo_view.cpu()
        delt_costh_emo = torch.zeros(costh_emo.size()).scatter_(1, label_emo_view, self.m)
        delt_costh_emo = delt_costh_emo.cuda()
        costh_m_emo = costh_emo - delt_costh_emo
        costh_m_s_emo = self.s * costh_m_emo
        loss_emo = self.ce(costh_m_s_emo, label_emo)
        logits_emo = costh_m_s_emo

        costh_spk = net_output["costh_spk"]
        targets_spk = sample['target_spk'].view(-1).long() - 4


        label_spk = targets_spk
        label_spk_view = label_spk.view(-1, 1)
        if label_spk_view.is_cuda: label_spk_view = label_spk_view.cpu()
        delt_costh_spk = torch.zeros(costh_spk.size()).scatter_(1, label_spk_view, self.m)
        delt_costh_spk = delt_costh_spk.cuda()
        costh_m_spk = costh_spk - delt_costh_spk
        costh_m_s_spk = self.s * costh_m_spk
        loss_spk = self.ce(costh_m_s_spk, label_spk)
        logits_spk = costh_m_s_spk


        loss = loss_spk + loss_emo

        target_lengths = sample["target_lengths"]
        ntokens = (
                sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
                )
        sample_size = sample['target_spk'].size(0) if self.sentence_avg else ntokens
        logging_output = {
            'loss': loss.data,
            #'ntokens': sample['ntokens'],
            'ntokens': ntokens,
            'nsentences': sample['target_spk'].size(0),
            'sample_size': sample_size,
        }
        preds_spk = logits_spk.argmax(dim=1)
        logging_output['ncorrect_spk'] = (preds_spk == targets_spk).sum()

        preds_emo = logits_emo.argmax(dim=1)
        logging_output['ncorrect_emo'] = (preds_emo == targets_emo).sum()
        return loss, sample_size, logging_output


    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))
        if len(logging_outputs) > 0 and 'ncorrect_emo' in logging_outputs[0]:
            ncorrect_emo = sum(log.get('ncorrect_emo', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_emo', 100.0 * ncorrect_emo / nsentences, nsentences, round=1)

        if len(logging_outputs) > 0 and 'ncorrect_spk' in logging_outputs[0]:
            ncorrect_spk = sum(log.get('ncorrect_spk', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy_spk', 100.0 * ncorrect_spk / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

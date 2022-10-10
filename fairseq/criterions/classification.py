# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion, AMSoftmax


@register_criterion('classification')
class ClassifyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        #self.criterion = AMSoftmax(768, 1211, margin=0.2, scale=30)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        #feats = net_output["features_out"]
        #feats = feats.transpose(0, 1)


        #padding_mask = net_output["encoder_padding_mask"]

        #feats = self.avg_pooling(feats, padding_mask)

        #targets = model.get_targets(sample, net_output).view(-1).long() - 4

        #loss, logits = self.criterion(feats, targets)
        #target_lengths = sample["target_lengths"]
        #ntokens = (
        #        sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        #        )
        #sample_size = sample['target'].size(0) if self.sentence_avg else ntokens #sample['ntokens']

        loss, ncorrect = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            #'ntokens': sample['ntokens'],
            'ntokens': sample['target'].size(0),
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        #preds = feats.argmax(dim=1)
        #logging_output['ncorrect'] = (preds == targets).sum()
        logging_output['ncorrect'] = ncorrect
        return loss, sample_size, logging_output


    def avg_pooling(self, feats, padding_mask):
        mask = 1 - padding_mask.int().unsqueeze(dim=2).repeat(1, 1, feats.size(2))
        feats = torch.div(torch.sum(feats.mul(mask), dim=1), torch.sum(mask, dim=1))

        return feats

    def linear_softmax_pooling(self, feats, padding_mask):
        mask = 1 - padding_mask.int().unsqueeze(dim=2).repeat(1, 1, feats.size(2))
        num = torch.sum(feats*feats*mask, dim=1)
        den = torch.sum(feats*mask, dim=1)
        return torch.div(num, den)


    def compute_loss_2(self, model, net_output, sample, reduce=True):

        padding_mask = net_output["encoder_padding_mask"]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        length = lprobs.size(0)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)[:,0]
        target = target.unsqueeze(dim=1).repeat(1, length)
        target = target.mul(1 - padding_mask.int())
        pad = 2*torch.ones_like(target).mul(padding_mask.int())
        target += pad
        target = target.view(-1)

        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    def compute_loss(self, model, net_output, sample, reduce=True):

        padding_mask = net_output["encoder_padding_mask"]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = torch.transpose(lprobs, 1, 0)
        mask = 1 - padding_mask.int().unsqueeze(dim=2).repeat(1, 1, lprobs.size(2))
        lprobs = torch.div(torch.sum(lprobs.mul(mask), dim=1), torch.sum(mask, dim=1))
        preds = lprobs.argmax(dim=1)
        targets = model.get_targets(sample, net_output)[:,0]
        targets = targets.view(-1)
        ncorrect = (preds == targets).sum()

        loss = F.nll_loss(
            lprobs,
            targets.long(),
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, ncorrect


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
        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

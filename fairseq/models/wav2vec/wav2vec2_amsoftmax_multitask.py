# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks, utils

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer

torch.autograd.set_detect_anomaly(True)

def add_common_args(parser):
    parser.add_argument("--w2v-path", help="path to wav2vec 2.0 model")
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help="if true, does not load pretrained weights",
    )
    parser.add_argument(
        "--dropout-input",
        type=float,
        metavar="D",
        help="dropout to apply to the input (after feat extr)",
    )
    parser.add_argument(
        "--final-dropout",
        type=float,
        metavar="D",
        help="dropout after transformer and before final projection",
    )
    parser.add_argument(
        "--apply-mask", action="store_true", help="apply masking during fine-tuning"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        metavar="D",
        help="dropout probability inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--attention-dropout",
        type=float,
        metavar="D",
        help="dropout probability for attention weights inside wav2vec 2.0 model",
    )
    parser.add_argument(
        "--activation-dropout",
        "--relu-dropout",
        type=float,
        metavar="D",
        help="dropout probability after activation in FFN inside wav2vec 2.0 model",
    )

    parser.add_argument(
        "--mask-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-prob", type=float, help="probability of replacing a token with mask"
    )

    parser.add_argument(
        "--mask-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--mask-channel-length", type=int, help="repeat the mask indices multiple times"
    )

    parser.add_argument(
        "--mask-channel-prob",
        type=float,
        help="probability of replacing a token with mask",
    )

    parser.add_argument(
        "--mask-channel-selection",
        type=str,
        choices=["static", "uniform", "normal", "poisson"],
        help="how to choose masks",
    )

    parser.add_argument(
        "--mask-channel-other",
        type=float,
        help="stdev of the mask length in case of 'normal' selection strategy",
    )

    parser.add_argument(
        "--no-mask-channel-overlap",
        action="store_true",
        help="whether to allow masks to overlap",
    )

    parser.add_argument(
        "--freeze-finetune-updates",
        default=0,
        type=int,
        help="dont finetune wav2vec for this many updates",
    )

    parser.add_argument(
        "--feature-grad-mult",
        default=None,
        type=float,
        help="reset feature grad mult in wav2vec 2.0 to this",
    )

    parser.add_argument(
        "--layerdrop",
        default=0.0,
        type=float,
        help="probability of dropping a layer in wav2vec 2.0",
    )


@register_model("wav2vec_amsoftmax_multitask")
class Wav2VecCtcMulti(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        add_common_args(parser)

    def __init__(self, w2v_encoder, args):
        super().__init__()
        self.w2v_encoder = w2v_encoder
        self.args = args

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        w2v_encoder = Wav2VecEncoder(args)
        return cls(w2v_encoder, args)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

    # def max_positions(self):
    #     return None



class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, args, tgt_dict=None):
        self.apply_mask = args.apply_mask

        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        if getattr(args, "w2v_args", None) is None:

            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.w2v_path, arg_overrides
            )
            w2v_args = state["args"]
        else:
            state = None
            w2v_args = args.w2v_args

        assert args.normalize == w2v_args.normalize, 'Fine-tuning works best when data normalization is the same'

        w2v_args.data = args.data
        task = tasks.setup_task(w2v_args)
        model = task.build_model(w2v_args)

        if state is not None and not args.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(args.final_dropout)
        self.freeze_finetune_updates = args.freeze_finetune_updates
        self.num_updates = 0
        #self.W = torch.nn.Parameter(torch.randn(768, len(tgt_dict)-4), requires_grad=True)

        self.W_spk = torch.nn.Parameter(torch.randn(768, 10), requires_grad=True)
        nn.init.xavier_normal_(self.W_spk, gain=1)

        self.W_emo = torch.nn.Parameter(torch.randn(768, 4), requires_grad=True)
        nn.init.xavier_normal_(self.W_emo, gain=1)
        #self.proj = Linear(d, len(tgt_dict))

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates


        with torch.no_grad() if not ft else contextlib.ExitStack():
            #x, padding_mask, inner_state = self.w2v_model.extract_features(**w2v_args)
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

        x = self.final_dropout(x)
        feats = x

        mask = 1 - padding_mask.int().unsqueeze(dim=2).repeat(1, 1, x.size(2))
        x = torch.div(torch.sum(x.mul(mask), dim=1), torch.sum(mask, dim=1))


        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_emo_norm = torch.norm(self.W_emo, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_emo_norm = torch.div(self.W_emo, w_emo_norm)
        costh_emo = torch.mm(x_norm, w_emo_norm)

        w_spk_norm = torch.norm(self.W_spk, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_spk_norm = torch.div(self.W_spk, w_spk_norm)
        costh_spk = torch.mm(x_norm, w_spk_norm)

        return {
            "encoder_out": x,# B x C
            "costh_emo": costh_emo,
            "costh_spk": costh_spk,
            "encoder_padding_mask": padding_mask,  # B x T
            "features_out":feats,
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)

        self.dropout = args.decoder_dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_embed_dim
        args.encoder_embed_dim = embed_dim

        self.layerdrop = args.decoder_layerdrop

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        args = copy.deepcopy(args)
        args.dropout = args.decoder_dropout
        args.attention_dropout = args.decoder_attention_dropout
        args.activation_dropout = args.decoder_activation_dropout

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        prev_output_tokens = prev_output_tokens.long()
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, attn, _ = layer(
                    x,
                    encoder_out["encoder_out"] if encoder_out is not None else None,
                    encoder_out["encoder_padding_mask"]
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    self_attn_mask=self.buffered_future_mask(x)
                    if incremental_state is None
                    else None,
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, {"attn": attn, "inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

class AM_softmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, **kwargs):
        super(AM_softmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)

        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        #prec1, _ = accuracy(costh_m_s.detach().cpu(), label.detach().cpu(), topk=(1, 5))
        return loss, costh_m_s




def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture("wav2vec_amsoftmax_multitask", "wav2vec_amsoftmax_multitask")
def base_architecture(args):
    args.no_pretrained_weights = getattr(args, "no_pretrained_weights", False)
    args.dropout_input = getattr(args, "dropout_input", 0)
    args.final_dropout = getattr(args, "final_dropout", 0)
    args.apply_mask = getattr(args, "apply_mask", False)
    args.dropout = getattr(args, "dropout", 0)
    args.attention_dropout = getattr(args, "attention_dropout", 0)
    args.activation_dropout = getattr(args, "activation_dropout", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.5)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0.5)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)

    args.freeze_finetune_updates = getattr(args, "freeze_finetune_updates", 0)
    args.feature_grad_mult = getattr(args, "feature_grad_mult", 0)
    args.layerdrop = getattr(args, "layerdrop", 0.0)



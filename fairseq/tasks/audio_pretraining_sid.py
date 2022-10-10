# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys

from fairseq.data import data_utils, FairseqDataset, iterators

from fairseq.data import FileAudioDatasetSid, Dictionary, AddTargetDatasetSid
from . import FairseqTask, register_task


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("audio_pretraining_sid")
class AudioPretrainingSidTask(FairseqTask):
    """

    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=int,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--max-sample-size",
            default=None,
            type=int,
            help="max sample size to crop to for batching. default = min sample length",
        )
        parser.add_argument(
            "--min-sample-size",
            default=None,
            type=int,
            help="min sample size to crop to for batching. default = same as --max-sample-size",
        )

        parser.add_argument(
            "--enable-padding",
            action="store_true",
            help="pad shorter samples instead of cropping",
        )

        parser.add_argument(
            "--labels",
            type=str,
            default=None,
            help="extension of the label file to load, if any",
        )

    def __init__(self, args, source_dictionary=None):
        super().__init__(args)
        self._target_dictionary = None
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        self.datasets[split] = FileAudioDatasetSid(
            manifest,
            sample_rate=self.args.sample_rate,
            max_sample_size=self.args.max_sample_size,
            min_sample_size=self.args.max_sample_size,
            min_length=self.args.min_sample_size,
            pad=self.args.labels is not None or self.args.enable_padding,
            normalize=self.args.normalize,
        )
        if self.args.labels:
            dict_path = os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
            self._target_dictionary = Dictionary.load(dict_path)
            label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDatasetSid(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=False,
            )

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)


    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        #if max_positions is not None:
        #    indices = self.filter_indices_by_size(
        #        indices, dataset, max_positions, ignore_invalid_inputs
        #    )

        # create mini-batches with given size constraints
  
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        
        # return a reusable, sharded iterator
        
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=getattr(self.args, 'data_buffer_size', 0),
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter


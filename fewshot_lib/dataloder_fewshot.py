import argparse
import os
from functools import partial
from typing import List

from torch.utils.data import DataLoader

from fewshot_lib import dataspec
from fewshot_lib.config import Split
from fewshot_lib.pipeline import worker_init_fn_, make_episode_pipeline, make_batch_pipeline


class DataConfig(object):
    """Common configuration options for creating data processing pipelines."""

    def __init__(
            self,
            args: argparse.Namespace
    ):
        """Initialize a DataConfig.
        """

        # General info
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle

        # Transforms and augmentations
        self.image_size = args.image_size
        self.test_transforms = args.test_transforms
        self.train_transforms = args.train_transforms


class EpisodeDescriptionConfig(object):
    """Configuration options for episode characteristics."""

    def __init__(self, args: argparse.Namespace):

        arg_groups = {
                'num_ways': (args.num_ways, ('min_ways', 'max_ways_upper_bound'), (args.min_ways, args.max_ways_upper_bound)),
                'num_query': (args.num_query, ('max_num_query',), (args.max_num_query,)),
                'num_support':
                        (args.num_support,  # noqa: E131
                        ('max_support_set_size', 'max_support_size_contrib_per_class',  # noqa: E128
                         'min_log_weight', 'max_log_weight'),
                        (args.max_support_set_size, args.max_support_size_contrib_per_class,  # noqa: E128
                         args.min_log_weight, args.max_log_weight)),
        }

        for first_arg_name, values in arg_groups.items():
            first_arg, required_arg_names, required_args = values
            if ((first_arg is None) and any(arg is None for arg in required_args)):
                # Get name of the nones
                none_arg_names = [
                        name for var, name in zip(required_args, required_arg_names)
                        if var is None
                ]
                raise RuntimeError(
                        'The following arguments: %s can not be None, since %s is None. '
                        'Please ensure the following arguments of EpisodeDescriptionConfig are set: '
                        '%s' % (none_arg_names, first_arg_name, none_arg_names))

        self.num_ways = args.num_ways
        self.num_support = args.num_support
        self.num_query = args.num_query
        self.min_ways = args.min_ways
        self.max_ways_upper_bound = args.max_ways_upper_bound
        self.max_num_query = args.max_num_query
        self.max_support_set_size = args.max_support_set_size
        self.max_support_size_contrib_per_class = args.max_support_size_contrib_per_class
        self.min_log_weight = args.min_log_weight
        self.max_log_weight = args.max_log_weight
        self.min_examples_in_class = args.min_examples_in_class
        self.ignore_bilevel_ontology = args.ignore_bilevel_ontology

    def max_ways(self):
        """Returns the way (maximum way if variable) of the episode."""
        return self.num_ways or self.max_ways_upper_bound


def get_fewshot_dataloader(opt: argparse.Namespace, sources: List[str], batch_size: int, split: dataspec.Split):
    data_config = DataConfig(opt)
    episod_config = EpisodeDescriptionConfig(opt)
    use_bilevel_ontology_list = [False] * len(sources)
    if episod_config.num_ways and len(sources) > 1:
        raise ValueError('For fixed episodes, not tested yet on > 1 data_lib')
    else:
        # Enable ontology aware sampling for breakhis
        if 'breakhis' in sources:
            use_bilevel_ontology_list[sources.index('breakhis')] = True
    episod_config.use_bilevel_ontology_list = use_bilevel_ontology_list

    all_dataset_specs = []
    for dataset_name in sources:
        dataset_records_path = os.path.join(data_config.data_path, dataset_name)
        dataset_spec = dataspec.load_dataset_spec(dataset_records_path)
        all_dataset_specs.append(dataset_spec)

    pipeline_fn = make_episode_pipeline

    dataset = pipeline_fn(dataset_spec_list=all_dataset_specs,
                          split=split,
                          data_config=data_config,
                          episode_descr_config=episod_config)

    worker_init_fn = partial(worker_init_fn_, seed=opt.seed)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers=opt.num_workers,
                        worker_init_fn=worker_init_fn)
    num_classes = sum([len(d_spec.get_classes(split=Split["TRAIN"])) for d_spec in all_dataset_specs])
    return loader, num_classes
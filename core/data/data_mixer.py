# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import json
import logging
import os
import re
import traceback
from typing import Any, Callable, Dict, Iterator, List, Optional, cast

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


def get_worker_info():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        num_workers = 1
        worker_id = 0
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id

    return worker_id, num_workers


def get_global_rank_info(rank, world_size):
    worker_id, num_workers = get_worker_info()
    dataloader_rank = rank * num_workers + worker_id
    dataloader_world_size = world_size * num_workers
    return dataloader_rank, dataloader_world_size


class JSONLIterator:
    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.f = open(fpath, "r", encoding="utf-8")
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank
        self.line_num = 0
        self.iter = iter(self.gen(infinite))
        self.iter_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self, infinite: bool) -> Iterator[Dict]:
        while True:
            if self.world_rank == 0:
                logger.info(f"Starting iteration {self.iter_id} over {self.fpath} ...")
            self.iter_id += 1
            while True:
                line, self.line_num = self.f.readline(), self.line_num + 1
                if not line:
                    break
                if (self.line_num - 1) % self.world_size == self.world_rank:
                    yield json.loads(line)
            if not infinite:
                break
            self.set_position(None)
        self.f.close()

    def set_position(self, position: Optional[int]):
        logger.warning(
            f"Setting JSONL position on {self.fpath} "
            f"({self.world_rank}/{self.world_size}): {position}"
        )
        if position is None:
            self.f.seek(0)
            self.line_num = 0
        else:
            assert isinstance(position, int)
            self.f.seek(position)
            self.line_num = (
                self.world_rank + 1
            )  # Restore value of line_num (modulo world_size)

    def get_position(self) -> Optional[int]:
        file_pos = self.f.tell()
        if file_pos == 0 and self.line_num == 0:
            return None
        assert (self.line_num - 1) % self.world_size == self.world_rank
        return file_pos

    def get_example_file(self):
        """
        Return the path to a sample file to infer the content key
        """
        return self.fpath

    def get_id(self):
        """
        Return an identifier for the dataset this iterator represents
        """
        return self.fpath


class JSONLDirectoryIterator:
    """
    The JSONLDirectoryIterator is a data wrapper around a dataset folder, which contains
    multiple JSONL files. Internally, it reuses the JSONLIterator class to iterate through
    each individual file, and then wraps onto the next file once the current one is exhausted.

    Once all files in the directory have been iterated over, we wrap back to the first file
    ( if infinite is true ).

    This enables us to iterate over a dataset one chunk at a time.

    Also, note that we open the next chunk file on an ondemand basis, which means that we can
    modify chunks mid training as well to add more data, fix issues, etc.
    """

    def __init__(
        self,
        dirpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.dirpath = dirpath
        self.world_size = world_size
        self.world_rank = world_rank

        fnames = [
            x
            for x in os.listdir(self.dirpath)
            if re.fullmatch(r".*chunk\.\d+.*\.jsonl", x)
        ]
        self.fpaths = [os.path.join(self.dirpath, fname) for fname in sorted(fnames)]
        assert (
            len(self.fpaths) > 0
        ), f"Specified dataset location {self.dirpath} is empty."

        # Generator for cycling through the list of files
        if infinite:
            self.fpaths_generator = cast(Iterator[str], itertools.cycle(self.fpaths))
        else:
            self.fpaths_generator = cast(Iterator[str], iter(self.fpaths))

        self.iter = iter(self.gen(infinite))
        self.jsonl_iterator: Optional[JSONLIterator] = None

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self, infinite: bool) -> Iterator[Dict]:
        # Handle the case when we're reloading from a saved state.
        if self.jsonl_iterator is not None:
            yield from self.jsonl_iterator

        for fpath in self.fpaths_generator:
            # Note that we set infinite to false here, because JSONLDirectoryIterator would take care of infinite looping
            self.jsonl_iterator = JSONLIterator(
                fpath,
                world_size=self.world_size,
                world_rank=self.world_rank,
                infinite=False,
            )

            yield from self.jsonl_iterator

    def set_position(self, state: Dict[str, Any]):
        logger.warning(
            f"Setting JSONL position on {self.dirpath} "
            f"({self.world_rank}/{self.world_size}): {state}"
        )
        fpath: Optional[str] = state["fpath"]
        position: Optional[int] = state["position"]
        if fpath is None or position is None:
            return

        assert isinstance(fpath, str)
        assert isinstance(position, int)

        # Fast forward the generator
        for fpath_candidate in self.fpaths_generator:
            if fpath_candidate == fpath:
                break

        # Create the JSONL iterator and set it's position appropriately
        self.jsonl_iterator = JSONLIterator(
            fpath,
            world_size=self.world_size,
            world_rank=self.world_rank,
            infinite=False,
        )
        self.jsonl_iterator.set_position(position)

    def get_position(self):
        if self.jsonl_iterator is None:
            return {
                "fpath": None,
                "position": None,
            }
        return {
            "fpath": self.jsonl_iterator.fpath,
            "position": self.jsonl_iterator.get_position(),
        }

    def get_example_file(self):
        """
        Return the path to a sample file to infer the content key
        """
        return self.fpaths[0]

    def get_id(self):
        """
        Return an identifier for the dataset this iterator represents
        """
        return self.dirpath


class IterativeJSONLDataset(IterableDataset):
    def __init__(
        self,
        global_rank: int,
        world_size: int,
        dataset_name: str,
        seed: int = 0,
        dataset_configs: Dict[str, Any] = {},
    ):
        self._dataset_name = dataset_name
        self._seed = seed
        self._dataset_conf = dataset_configs[dataset_name]

        self.global_rank = global_rank
        self.world_size = world_size
        self.data_path = self._dataset_conf.annotation

    def worker_init(self, worker_id, num_workers):
        dataloader_rank = self.global_rank * num_workers + worker_id
        dataloader_world_size = self.world_size * num_workers
        if os.path.isfile(self.data_path):
            self.jsonl_iterator = JSONLIterator(
                self.data_path,
                world_size=dataloader_world_size,
                world_rank=dataloader_rank,
                infinite=True,
            )
        else:
            self.jsonl_iterator = JSONLDirectoryIterator(
                dirpath=self.data_path,
                world_size=dataloader_world_size,
                world_rank=dataloader_rank,
                infinite=True,
            )
        if worker_id == 0:
            logger.info(
                f"Initializing JSONLDataset {self._dataset_name} on "
                f"dataloader rank {dataloader_rank} and world size {dataloader_world_size}"
            )

    def state_dict(self):
        pos = self.jsonl_iterator.get_position()
        if isinstance(pos, Dict):
            return pos
        else:
            return {"single_jsonl_position": pos}

    def load_state_dict(self, state_dict):
        if "single_jsonl_position" in state_dict:
            self.jsonl_iterator.set_position(state_dict["single_jsonl_position"])
        else:
            self.jsonl_iterator.set_position(state_dict)
        logger.info(f"JSONLDataset {self._dataset_name} resuming from {state_dict}.")

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.jsonl_iterator)


class DatasetMixer(IterableDataset):
    def __init__(
        self,
        mix: str,
        global_rank: int,
        world_size: int,
        seed: int = 0,
        preprocessors: List[Callable] = [],
        dataset_configs: Dict[str, Any] = {},
    ):
        super().__init__()

        self.dataset_and_preprocessors = []
        self.weights = []
        self.dataset_names = []
        self.totals = []

        self.global_rank = global_rank
        self.world_size = world_size
        self.seed = seed

        mix = "".join(mix.split())  # Remove whitespace

        for elem in mix.split(","):
            ds, weight = elem.split(":")

            if ds not in dataset_configs:
                raise ValueError(f"Dataset {ds} not found in dataset_configs.")
            if ds in self.dataset_names:
                raise ValueError(
                    f"Dataset {ds} already in the mix. Each dataset can only be used once."
                )

            dataset = IterativeJSONLDataset(
                global_rank=global_rank,
                world_size=world_size,
                dataset_name=ds,
                seed=seed,
                dataset_configs=dataset_configs,
            )
            _preprocessors = [
                p(dataset_config=dataset_configs[ds]) for p in preprocessors
            ]

            self.dataset_and_preprocessors.append((dataset, _preprocessors))
            self.weights.append(float(weight))
            self.dataset_names.append(ds)
            self.totals.append(0)

        self.weights = [w / sum(self.weights) for w in self.weights]
        self.rng = None

    def state_dict(self):
        return {
            "datasets": {
                ds_name: ds.state_dict()
                for ds_name, (ds, _) in zip(
                    self.dataset_names, self.dataset_and_preprocessors
                )
            },
            "totals": {
                ds_name: total
                for ds_name, total in zip(self.dataset_names, self.totals)
            },
            "rng": (
                [
                    s.tolist() if isinstance(s, np.ndarray) else s
                    for s in self.rng.get_state()
                ]
                if self.rng is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict):
        for ds_name, sd in state_dict["datasets"].items():
            if ds_name in self.dataset_names:
                ds_idx = self.dataset_names.index(ds_name)
                ds, _ = self.dataset_and_preprocessors[ds_idx]
                ds.load_state_dict(sd)
                self.totals[ds_idx] = state_dict["totals"][ds_name]

        logger.info(
            f"DatasetMixer with datasets {self.dataset_names} resuming with total samples seen {self.totals} on process {os.getpid()}."
        )

        if state_dict["rng"] is not None:
            self.rng = np.random.RandomState()
            rng_state = [
                np.array(s) if isinstance(s, list) else s for s in state_dict["rng"]
            ]
            self.rng.set_state(rng_state)

    def worker_init(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        for dataset, _ in self.dataset_and_preprocessors:
            if hasattr(dataset, "worker_init"):
                dataset.worker_init(worker_id, worker_info.num_workers)

    def __iter__(self):
        if self.rng is None:
            rank, world_size = get_global_rank_info(self.global_rank, self.world_size)
            self.rng = np.random.RandomState((rank, world_size, self.seed))

        while True:
            try:
                src_id = self.rng.choice(len(self.weights), p=self.weights)
                dataset, preprocessors = self.dataset_and_preprocessors[src_id]
                out = next(dataset)
                for preprocessor in preprocessors:
                    if out is not None:
                        out = preprocessor(out, self.rng)

                if out is None:
                    continue

                self.totals[src_id] += 1
                yield out
            except Exception as e:
                logger.error(
                    f"Error while iterating over dataset {self.dataset_names[src_id]}: {e}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )


class PersistentDataLoader:
    """
    A _very_ persistent dataloader.

    Uses StatefulDataLoader to save dataset state (make sure dataset has a state_dict() and load_state_dict() method).
    Also keeps the dataloader iterator and the epoch iterator separate, so that the dataloader workers are persistent.

    Also laughs in the face of torch when it tries to kill the whole job because a worker died. Instead, this dataloader
    will just gracefully restart the underlying iterator and corresponding workers, while additionally loading the state dict
    so that it resumes from where it left off.

    This may or may not be a good idea.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        workers,
        collate_fn=None,
        positions=None,
    ):
        from torchdata.stateful_dataloader import StatefulDataLoader

        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            # pin_memory=True,
            multiprocessing_context="fork" if workers > 0 else None,
            collate_fn=collate_fn,
            worker_init_fn=(
                dataset.worker_init if hasattr(dataset, "worker_init") else None
            ),
            # persistent_workers=(workers > 0),
            snapshot_every_n_steps=1,
        )

        if positions is not None:
            self.load_state_dict(positions)

        self._dataloader_iter = iter(self.dataloader)

        # # Stop torch from killing us all
        # register_subscriber(self)

    def state_dict(self):
        return self.dataloader.state_dict()

    def load_state_dict(self, state_dict):
        self.dataloader.load_state_dict(state_dict)

    def __del__(self):
        pass  # unregister_subscriber(self)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.iter = self.gen()
        return self

    def __next__(self):
        return next(self.iter)

    def _refresh_iter(self):
        # Called by the signal handler when a worker dies
        self._dataloader_iter = None

    def _get_next_sample(self):
        if self._dataloader_iter is None:
            self.dataloader.load_state_dict(self.dataloader.state_dict())
            self._dataloader_iter = iter(self.dataloader)

        try:
            return next(self._dataloader_iter)
        except (KeyboardInterrupt, StopIteration):
            raise
        except Exception as e:
            if self._dataloader_iter is None:
                # An interrupt forced us to respawn the dataloaders, do it next sample
                return self._get_next_sample()
            else:
                raise e

    def gen(self):
        while True:
            try:
                yield self._get_next_sample()
            except StopIteration:
                raise

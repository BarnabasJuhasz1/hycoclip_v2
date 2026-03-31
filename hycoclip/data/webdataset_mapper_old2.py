from __future__ import annotations

import copy
import glob
import random
import numpy as np
from typing import Callable

import json
from pathlib import Path
from collections import OrderedDict
import webdataset as wds
import wordsegment as ws
from loguru import logger
from torch.utils.data import IterableDataset
from torchvision import transforms as T

import hycoclip.utils.distributed as dist

import json
import torch

ws.load()


def check_parent_keys(sample):
    """
    Check if all parent keys are present in the sample.
    """
    if "numparents.txt" not in sample:
        return False
    num_of_parents = int(sample["numparents.txt"])
    for box in range(num_of_parents):
        if f"parent{box:03d}.txt" not in sample or f"parent{box:03d}.jpg" not in sample:
            return False
    return True
    # parents_diff_child = sum((len(sample[f"parent{box:03d}.txt"]) > len(sample["child.txt"])-2) for box in range(num_of_parents))
    # return parents_diff_child == num_of_parents
    

class JsonAnnotationLoader:
    def __init__(self, folder, annotation_key, max_cache_size=10):
        self.cache = OrderedDict()
        self.folder = Path(folder)
        self.max_cache_size = max_cache_size
        self.annotation_key = annotation_key

    def __call__(self, sample):
        shard_path = sample["__url__"]
        shard_id = Path(shard_path).stem

        if shard_id not in self.cache:
            json_path = shard_id + ".json"
            json_path = self.folder / json_path
            with open(json_path, "r") as f:
                annotations = json.load(f)

            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)

            self.cache[shard_id] = annotations

        else:
            # Mark as recently used
            self.cache.move_to_end(shard_id)

        sample[self.annotation_key] = self.cache[shard_id][sample['__key__']]
        return sample

class ImageTextWebDataset(IterableDataset):
    """
    Iterable dataset that serves instances from a lot of TAR file shards.
    This class uses `WebDataset <https://github.com/webdataset/webdataset>`_
    internally, and expects TAR files to be arranged in a compatible format.
    """

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0
    ):
        """
        Args:
            tarfiles: Path(s) or glob-patterns for TAR files in WebDataset format.
            mapper: A callable to transform a single dataset dict (image and
                annotations). May implement data augmentation and tokenization.
            buffer_size: Size of the internal buffer of instances. Data is read
                sequentially from TAR files into this buffer and served randomly.
                Shuffling will be disabled if this is set to zero.
            infinite_stream: Yield an infinite stream of instances if this is
                True. In such cases, the user must terminate this iterator manually
                (e.g. run a fixed sized for-loop in training code).
            seed: Random seed for buffer shuffling. If provided, this dataloader
                will load batches deterministically across different runs (only if
                batch size and number of GPUs/CPUs are same). This seed can either
                be same or different per GPU process for multi-GPU training.
        """
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        # Convert a single path (glob) to a list.
        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        # Expand all glob patterns to list a full list of individual TAR files.
        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        # Sort paths; webdataset performs a deterministic shuffle (internally).
        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        # Shard the TAR file paths as per number of GPU processes to avoid loading
        # duplicates.
        _rank, _world_size = dist.get_rank(), dist.get_world_size()
        self.tarfiles = self.tarfiles[_rank::_world_size]
        logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng),
            )

        # Decode images using PIL and apply custom mapper.
        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.select(check_parent_keys))  # Ensure all parent keys are present.
        pipeline.append(wds.map(self.mapper))
        # If a mapper returns a list of samples (to emit multiple samples per input),
        # flatten them into individual samples here.
        pipeline.append(wds.flatmap(lambda x: x if isinstance(x, list) else [x]))

        if self.infinite_stream:
            # Sample an infinite stream of dataset dicts.
            while True:
                pipeline_copy = copy.deepcopy(pipeline)
                yield from pipeline_copy
        else:
            # Run for one epoch and stop:
            yield from pipeline


class GroundedDatasetTarMapper:
    """
    Mapper to pre-process image-text instances from Grounded dataset TAR files.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
        """
        self.image_transform = T.Compose(image_transform)

    def __call__(self, dataset_dict: dict):
        num_boxes = int(dataset_dict["numparents.txt"])
        random_box = random.randrange(num_boxes)
        return {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
            "box_image": self.image_transform(dataset_dict[f"parent{random_box:03d}.jpg"]),
            "box_text": dataset_dict[f"parent{random_box:03d}.txt"],
        }
        
        
class ExtendedGroundedDatasetTarMapper:
    """
    Mapper to pre-process image-text instances from Grounded dataset TAR files.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
        """
        self.image_transform = T.Compose(image_transform)
        self.annotation_loader = JsonAnnotationLoader(
            folder="datasets/HyperHi5/v1/json_pairwise",
            annotation_key="text_hierarchy",
            max_cache_size=100
        )

    def __call__(self, dataset_dict: dict):
        num_boxes = int(dataset_dict["numparents.txt"])
        random_box = random.randrange(num_boxes)
        dataset_dict = self.annotation_loader(dataset_dict)

        parent_id = f"parent{random_box:03d}"
        parent_original = dataset_dict["text_hierarchy"][parent_id]["original"]
        # skip the first element of hierarchy because it is equal to the parent box text
        hierarchy_list = parent_original.get("hierarchy", [])[1:]

        orig = {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
            "box_image": self.image_transform(dataset_dict[f"{parent_id}.jpg"]),
            "box_text": dataset_dict[f"{parent_id}.txt"],
            "text_hierarchy": hierarchy_list,
            "scores": np.array([parent_original["pairwise_scores"][str(i)]["final_score"]
                       for i in range(len(parent_original["pairwise_scores"]))], dtype=np.float32),
        }

        if not self.add_hierarchy_extra:
            return orig

        # Create an extra sample where box_text is a random element from the hierarchy
        if hierarchy_list:
            if self.extra_seed is not None:
                rng = random.Random(self.extra_seed + (hash(dataset_dict.get("__key__", "")) & 0xFFFFFFFF))
                new_box_text = rng.choice(hierarchy_list)
            else:
                new_box_text = random.choice(hierarchy_list)
        else:
            new_box_text = orig["box_text"]

        extra = copy.deepcopy(orig)
        extra["box_text"] = new_box_text
        extra["is_extra"] = True

        return [orig, extra]




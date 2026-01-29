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
        return {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
            "box_image": self.image_transform(dataset_dict[f"parent{random_box:03d}.jpg"]),
            "box_text": dataset_dict[f"parent{random_box:03d}.txt"],
            "text_hierarchy": dataset_dict["text_hierarchy"][f"parent{random_box:03d}"]["original"]["hierarchy"][1:], # the first element is equal to box_text
            "scores": np.array([dataset_dict["text_hierarchy"][f"parent{random_box:03d}"]["original"]["pairwise_scores"][str(i)]["final_score"]
                       for i in range(len(dataset_dict["text_hierarchy"][f"parent{random_box:03d}"]["original"]["pairwise_scores"]))], dtype=np.float32),
        }



class GroundedDatasetTarMapperV2:
    """
    Mapper to pre-process image-text instances from Grounded dataset TAR files.
    """

    def __init__(
        self,
        hierfiles: str | list[str],
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

        # Load hierarchy JSONs
        if isinstance(hierfiles, str):
            hierfiles = [hierfiles]

        self.key_to_file = {}  # Only maps key -> JSON path
        self.json_cache = {}   # Optional: in-memory LRU-style cache (can limit size)

        self.hierfile_count = 0
        for path in hierfiles:
            for file in glob.glob(path):
                with open(file, "r") as f:
                    json_data = json.load(f)
                    for key in json_data:
                        self.key_to_file[key] = file
                self.hierfile_count += 1
                
        logger.info(f"{self.__class__.__name__} found {self.hierfile_count} JSONs and {len(self.key_to_file)} keys.")

        self.image_transform = T.Compose(image_transform)

    def __call__(self, dataset_dict: dict):
        key = dataset_dict["__key__"]
        # num_boxes = int(dataset_dict["numparents.txt"])
        # random_box = random.randrange(num_boxes)
        # parent_id = f"parent{random_box:03d}"
        # print(f"length of key-to-file mapping dict: {len(self.key_to_file)}")
        
        hierarchy = None
        json_path = self.key_to_file.get(key, None)
        if json_path is not None:
            if json_path not in self.json_cache:
                try:
                    with open(json_path, "r") as f:
                        self.json_cache[json_path] = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load JSON {json_path}: {e}")
                    self.json_cache[json_path] = {}
            json_data = self.json_cache[json_path]
            key_content = json_data.get(key, {})
            assert(key_content != None and key_content != {}), f"KEY CONTENT IS NONE: PATH:{json_path}, KEY:{key}, NUM_BOXES:{num_boxes}, RANDOM BOX:{random_box_id}, PARENT_ID:{parent_id}"
            # print(f"RANDOM BOX: {random_box_id}, PARENT KEYS: {list(key_content.keys())}")
            non_empty_parent_ids = list(_parent_id for _parent_id in key_content.keys() if key_content.get(_parent_id) != {} and isinstance(key_content.get(_parent_id), dict))
            # non_empty_parent = 0
            # parent_id = 0
            # for _parent_id in list(key_content.keys()):
            #     if key_content.get(_parent_id) != {}:

            #         if (non_empty_parent == random_box_id):
            #             parent_id = _parent_id
            #             break
            #         non_empty_parent += 1
            
            # assert(random_box_id < len(list(key_content.keys()))), f"ERROR WITH PATH:{json_path} KEY:{key} NUM_BOXES:{num_boxes}, RANDOM BOX: {random_box_id} PARENT_ID:{parent_id} PARENT KEYS: {list(key_content.keys())}"
            # parent_id = list(key_content.keys())[random_box_id]

            num_boxes = len(non_empty_parent_ids)
            random_box_id = random.randrange(num_boxes)
            parent_id = non_empty_parent_ids[random_box_id]
            parent_content_hier = key_content.get(parent_id, None)
            if parent_content_hier and parent_content_hier != {} and isinstance(parent_content_hier, dict):
                complex_hierarchy = parent_content_hier.get("original", None)
                if complex_hierarchy:
                    hierarchy = complex_hierarchy.get("hierarchy", None)
                    #pairwise_entail_scores = [complex_hierarchy.get("pairwise_scores").get(str(i)).get("final_score") for i in range(4)]
                    pairwise_entail_scores = torch.tensor([
                        complex_hierarchy["pairwise_scores"][str(i)]["final_score"]
                        for i in range(4)
                    ])
                    # print(f"path: {json_path}, for hier: {key} {parent_id}, the scores are {pairwise_entail_scores}")

        else:
            print(f"json_path IS NONE: {json_path} BECAUSE KEY: {key}")
        # if hierarchy == None:
            # logger.info(f"{self.__class__.__name__} HIERARCHY IS NONE: PATH:{json_path}, KEY:{key}, NUM_BOXES:{num_boxes}, RANDOM BOX:{random_box}, PARENT_ID:{parent_id}")
        assert(hierarchy != None), f"HIERARCHY IS NONE: PATH:{json_path}, KEY:{key}, NUM_BOXES:{num_boxes}, RANDOM BOX:{random_box_id}, PARENT_ID:{parent_id}"

        assert(len(hierarchy) == 5), f"Hierarchy length != 5: {hierarchy}"
        return {
            "__key__": key,
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
            "box_image": self.image_transform(dataset_dict[f"{parent_id}.jpg"]),
            "box_text": dataset_dict[f"{parent_id}.txt"],
            "hierarchy": hierarchy,
            "pairwise_scores": pairwise_entail_scores,
        }
    
            

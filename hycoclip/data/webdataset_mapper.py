from __future__ import annotations

import glob
import random
import tarfile as tarfile_module
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


def _grouped_tar_to_samples(src, handler=wds.warn_and_continue):
    """Read tar files and yield samples grouped by key.

    Unlike wds.tarfile_to_samples which requires consecutive entries per key,
    this accumulates ALL entries with the same key from the entire tar before
    yielding — needed when files are stored interleaved rather than grouped.
    """
    for shard in src:
        url = shard["url"] if isinstance(shard, dict) and "url" in shard else shard

        samples = {}
        try:
            with tarfile_module.open(url) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    dot_idx = name.find('.')
                    if dot_idx == -1:
                        continue
                    key = name[:dot_idx]
                    ext = name[dot_idx + 1:]
                    f = tf.extractfile(member)
                    if f is not None:
                        if key not in samples:
                            samples[key] = {'__key__': key, '__url__': url}
                        samples[key][ext] = f.read()
        except Exception as e:
            try:
                handler(e)
            except StopIteration:
                return
            continue

        yield from samples.values()


grouped_tar_to_samples = wds.pipelinefilter(_grouped_tar_to_samples)


def check_parent_keys(sample):
    """
    Check if all parent keys are present in the sample (post-decode, values are strings/PIL).
    """
    if "numparents.txt" not in sample:
        return False
    num_of_parents = int(sample["numparents.txt"])
    for box in range(num_of_parents):
        if f"parent{box:03d}.txt" not in sample or f"parent{box:03d}.jpg" not in sample:
            return False
    return True


def check_parent_keys_raw(sample):
    """
    Check if all parent keys are present before decoding (values are raw bytes).
    Avoids wasting CPU on BICUBIC decoding for samples that will be filtered.
    """
    if "numparents.txt" not in sample:
        return False
    try:
        num_of_parents = int(sample["numparents.txt"].decode().strip())
    except (ValueError, UnicodeDecodeError, AttributeError):
        return False
    for box in range(num_of_parents):
        if f"parent{box:03d}.txt" not in sample or f"parent{box:03d}.jpg" not in sample:
            return False
    return True


def check_child_keys_raw(sample):
    """
    Check if the child image/caption keys are present, ignoring any parent
    (box) keys. Used to read GRIT-format tars without requiring boxes.
    """
    return "child.jpg" in sample and "child.txt" in sample



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
            try:
                with open(json_path, "r") as f:
                    annotations = json.load(f)
            except FileNotFoundError:
                # A few TAR shards have no annotation JSON at all. They are still
                # usable training data, so cache an empty mapping and let the
                # caller fall back rather than killing the worker.
                logger.warning(f"No annotation JSON at {json_path}; shard will be served unannotated.")
                annotations = {}

            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)

            self.cache[shard_id] = annotations

        else:
            # Mark as recently used
            self.cache.move_to_end(shard_id)

        # Only a fraction of the TAR keys carry an annotation; missing ones get
        # `None` and are handled by the mapper.
        sample[self.annotation_key] = self.cache[shard_id].get(sample["__key__"])
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
        initial_buffer_size: int | None = None,
        infinite_stream: bool = True,
        seed: int = 0,
        sample_filter: Callable = check_parent_keys_raw,
    ):
        """
        Args:
            tarfiles: Path(s) or glob-patterns for TAR files in WebDataset format.
            mapper: A callable to transform a single dataset dict (image and
                annotations). May implement data augmentation and tokenization.
            buffer_size: Size of the internal shuffle buffer. Data is read
                sequentially from TAR files into this buffer and served randomly.
                Shuffling will be disabled if this is set to zero.
            initial_buffer_size: How many samples must be loaded into the shuffle
                buffer before the pipeline starts yielding. Defaults to buffer_size
                (fill entirely before starting). Set lower (e.g. 500) to reduce
                startup latency at a minor cost to shuffle quality at the very start.
            infinite_stream: Yield an infinite stream of instances if this is
                True. In such cases, the user must terminate this iterator manually
                (e.g. run a fixed sized for-loop in training code).
            seed: Random seed for buffer shuffling. If provided, this dataloader
                will load batches deterministically across different runs (only if
                batch size and number of GPUs/CPUs are same). This seed can either
                be same or different per GPU process for multi-GPU training.
            sample_filter: Predicate applied to raw (undecoded) samples to decide
                which ones survive. Defaults to requiring valid parent/box keys;
                pass `check_child_keys_raw` to only require child.jpg/child.txt
                (e.g. for using GRIT tars without boxes, alongside a mapper that
                doesn't read parent fields).
        """
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.initial_buffer_size = initial_buffer_size if initial_buffer_size is not None else buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed
        self.sample_filter = sample_filter

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

        # Shard the TAR file paths by count (simpler and avoids I/O contention).
        # Getting file sizes from network storage causes all ranks to compete for I/O.
        _rank, _world_size = dist.get_rank(), dist.get_world_size()
        
        if _world_size > 1:
            # Simple count-based sharding: distribute files evenly by count
            files_per_rank = (len(self.tarfiles) + _world_size - 1) // _world_size
            start_idx = _rank * files_per_rank
            end_idx = min(start_idx + files_per_rank, len(self.tarfiles))
            self.tarfiles = self.tarfiles[start_idx:end_idx]
            logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")
        else:
            logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            grouped_tar_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.initial_buffer_size, rng=rng),
            )

        # Filter on raw bytes before decoding so invalid samples never get decoded.
        pipeline.append(wds.select(self.sample_filter))
        # Decode images using PIL and apply custom mapper.
        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper))

        if self.infinite_stream:
            # Sample an infinite stream of dataset dicts.
            # Each call to `yield from pipeline` invokes pipeline.__iter__() fresh,
            # restarting from the beginning — no deepcopy needed.
            while True:
                yield from pipeline
        else:
            # Run for one epoch and stop:
            yield from pipeline


class ChildOnlyTarMapper:
    """
    Mapper that reads only the child image/caption from GRIT-format TAR files,
    ignoring any parent (box) keys. Pairs with `ImageTextWebDataset(sample_filter=
    check_child_keys_raw)` to use GRIT data for box-agnostic training (e.g. plain
    CLIP/MERU), producing the same {"image", "text"} shape as `ImageTextTarMapper`
    so it can be mixed with flat (non-GRIT) datasets in the same batch.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
    ):
        self.image_transform = T.Compose(image_transform)

    def __call__(self, dataset_dict: dict):
        return {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
        }


class FlatImageTextWebDataset(IterableDataset):
    """
    Iterable dataset for flat (image, caption) TAR shards with no grounding
    boxes, e.g. img2dataset output (`<key>.jpg` / `<key>.txt` / `<key>.json`
    per sample, stored consecutively). Unlike `ImageTextWebDataset`, this uses
    webdataset's standard `tarfile_to_samples` reader instead of the grouped
    reader, since there is no parent/child structure here.
    """

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        initial_buffer_size: int | None = None,
        infinite_stream: bool = True,
        seed: int = 0
    ):
        """
        Args: same as `ImageTextWebDataset`.
        """
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.initial_buffer_size = initial_buffer_size if initial_buffer_size is not None else buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        _rank, _world_size = dist.get_rank(), dist.get_world_size()

        if _world_size > 1:
            files_per_rank = (len(self.tarfiles) + _world_size - 1) // _world_size
            start_idx = _rank * files_per_rank
            end_idx = min(start_idx + files_per_rank, len(self.tarfiles))
            self.tarfiles = self.tarfiles[start_idx:end_idx]
            logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")
        else:
            logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.initial_buffer_size, rng=rng),
            )

        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper))

        if self.infinite_stream:
            while True:
                yield from pipeline
        else:
            yield from pipeline


class ImageTextTarMapper:
    """
    Mapper to pre-process image-text instances from flat (no grounding boxes)
    TAR files, e.g. img2dataset output. Pairs with `FlatImageTextWebDataset`.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
    ):
        self.image_transform = T.Compose(image_transform)

    def __call__(self, dataset_dict: dict):
        return {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["jpg"]),
            "text": dataset_dict["txt"],
        }


class MixedIterableDataset(IterableDataset):
    """
    Combines multiple IterableDatasets, sampling from each independently at
    every step according to fixed weights (e.g. 90% GRIT, 10% pixmo-cap).
    All sub-datasets must yield dicts with the same keys.
    """

    def __init__(self, datasets: list[IterableDataset], weights: list[float], seed: int = 0):
        """
        Args:
            datasets: Sub-datasets to sample from (each should be an
                infinite-stream IterableDataset, e.g. `ImageTextWebDataset`
                or `FlatImageTextWebDataset`).
            weights: Sampling weight per dataset, in the same order. Need not
                sum to 1 (normalized internally).
            seed: Random seed for the mixing choice. Combined with each
                worker's id so DataLoader workers don't all make the same
                sequence of choices.
        """
        super().__init__()
        assert len(datasets) == len(weights)
        total = sum(weights)
        self.datasets = datasets
        self.weights = [w / total for w in weights]
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed if worker_info is None else self.seed + worker_info.id
        rng = random.Random(worker_seed)

        iterators = [iter(d) for d in self.datasets]
        indices = list(range(len(self.datasets)))
        while True:
            idx = rng.choices(indices, weights=self.weights, k=1)[0]
            yield next(iterators[idx])


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
        use_proposed_hierachies: bool = False,
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
            use_proposed_hierachies: Accepted for config compatibility.
        """
        self.image_transform = T.Compose(image_transform)
        self.use_proposed_hierachies = use_proposed_hierachies

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
    Mapper to pre-process image-text instances from Grounded dataset TAR files,
    enriched with the HyperHi5 text-abstraction hierarchies.

    HyperHi5 only annotates a subset of GRIT (roughly 15% of keys, and not every
    box of an annotated key), so this mapper serves the *whole* of GRIT and falls
    back to repeating the box caption when a sample has no hierarchy. That
    fallback is deliberate rather than a placeholder: with `box_text` in every
    hierarchy slot, each hierarchy loss term collapses exactly onto a term the
    base HyCoCLIP loss already computes (box_text vs image, vs box_image, vs
    text), so unannotated samples keep training as ordinary HyCoCLIP instead of
    contributing a fabricated target.
    """

    # Number of hierarchy levels served per sample. The stored chains have 5
    # entries, the first of which duplicates the box caption and is dropped.
    _NUM_LEVELS = 4

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
        # use_extra_hierarchy_samples: bool = False,
        use_proposed_hierachies: bool = False,
        hier_folder: str = "datasets/train/HyperHi5/v1/json_pairwise",
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
            use_proposed_hierachies: Prefer the re-ordered `proposed` chain over
                `original` for the boxes that have one (about a third of them).
            hier_folder: Directory of per-shard hierarchy JSONs, named to match
                the TAR shards (`00000.json` for `00000.tar`).
        """
        self.image_transform = T.Compose(image_transform)
        self.annotation_loader = JsonAnnotationLoader(
            folder=hier_folder,
            annotation_key="text_hierarchy",
            max_cache_size=8
        )
        # self.use_extra_hierarchy_samples = use_extra_hierarchy_samples
        self.use_proposed_hierachies = use_proposed_hierachies

    def __call__(self, dataset_dict: dict):
        num_boxes = int(dataset_dict["numparents.txt"])
        random_box = random.randrange(num_boxes)
        dataset_dict = self.annotation_loader(dataset_dict)

        parent_id = f"parent{random_box:03d}"
        box_text = dataset_dict[f"{parent_id}.txt"]

        annotations = dataset_dict["text_hierarchy"]
        # The drawn box may be unannotated even when the key is.
        parent_annotation = annotations.get(parent_id) if annotations else None

        hierarchy_list, scores = None, None
        if parent_annotation:
            # decide whether to use proposed hierarchies or original ones
            if self.use_proposed_hierachies and "proposed" in parent_annotation:
                # use proposed if available, otherwise fallback to original
                parent_original = parent_annotation["proposed"]
            else:
                # use original hierarchies
                parent_original = parent_annotation["original"]

            # skip the first element of hierarchy because it is equal to the parent box text
            hierarchy = parent_original.get("hierarchy", [])[1:]
            pairwise = parent_original.get("pairwise_scores", {})
            # Chain length is uniform across the corpus, but a short entry would
            # otherwise break collation of the whole batch.
            if len(hierarchy) == self._NUM_LEVELS and len(pairwise) == self._NUM_LEVELS:
                hierarchy_list = hierarchy
                scores = np.array(
                    [pairwise[str(i)]["final_score"] for i in range(self._NUM_LEVELS)],
                    dtype=np.float32,
                )

        if hierarchy_list is None:
            # Unannotated: degenerate to the box caption (see class docstring).
            # Zero scores also neutralise the score-modulated terms in the
            # re-weight losses.
            hierarchy_list = [box_text] * self._NUM_LEVELS
            scores = np.zeros(self._NUM_LEVELS, dtype=np.float32)

        orig = {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
            "box_image": self.image_transform(dataset_dict[f"{parent_id}.jpg"]),
            "box_text": box_text,
            "text_hierarchy": hierarchy_list,
            "scores": scores,
        }

        # if not self.use_extra_hierarchy_samples:
        return orig

        # does not work, because the iterable expects a single dictionary not a list of dictionaries

        # # return a list of 5 samples for each element in hierarchy (original + 4 extra)
        # extra_hierarchies = [copy.deepcopy(orig) for _ in range(len(hierarchy_list))]
        # for i, extra in enumerate(extra_hierarchies):
        #     extra["box_text"] = hierarchy_list[i]

        # return [orig] + extra_hierarchies


class GroundedDatasetTarMapperV2:
    """
    Mapper to pre-process image-text instances from Grounded dataset TAR files with hierarchies.
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
            hierfiles: Path(s) or glob-patterns for hierarchy JSON files.
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
            
            non_empty_parent_ids = list(_parent_id for _parent_id in key_content.keys() if key_content.get(_parent_id) != {} and isinstance(key_content.get(_parent_id), dict))
            
            num_boxes = len(non_empty_parent_ids)
            random_box_id = random.randrange(num_boxes)
            parent_id = non_empty_parent_ids[random_box_id]
            parent_content_hier = key_content.get(parent_id, None)
            if parent_content_hier and parent_content_hier != {} and isinstance(parent_content_hier, dict):
                complex_hierarchy = parent_content_hier.get("original", None)
                if complex_hierarchy:
                    hierarchy = complex_hierarchy.get("hierarchy", None)
                    pairwise_entail_scores = torch.tensor([
                        complex_hierarchy["pairwise_scores"][str(i)]["final_score"]
                        for i in range(4)
                    ])

        if hierarchy is None:
            logger.warning(f"{self.__class__.__name__} HIERARCHY IS NONE for key: {key}")
            # Fallback: return a simple sample without hierarchy
            num_boxes = int(dataset_dict.get("numparents.txt", 1))
            random_box = random.randrange(num_boxes)
            return {
                "__key__": key,
                "image": self.image_transform(dataset_dict["child.jpg"]),
                "text": dataset_dict["child.txt"],
                "box_image": self.image_transform(dataset_dict[f"parent{random_box:03d}.jpg"]),
                "box_text": dataset_dict[f"parent{random_box:03d}.txt"],
                "hierarchy": [dataset_dict[f"parent{random_box:03d}.txt"]] * 5,
                "pairwise_scores": torch.ones(4),
            }

        return {
            "__key__": key,
            "image": self.image_transform(dataset_dict["child.jpg"]),
            "text": dataset_dict["child.txt"],
            "box_image": self.image_transform(dataset_dict[f"{parent_id}.jpg"]),
            "box_text": dataset_dict[f"{parent_id}.txt"],
            "hierarchy": hierarchy,
            "pairwise_scores": pairwise_entail_scores,
        }





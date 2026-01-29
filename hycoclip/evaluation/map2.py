
import argparse

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger

from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager

from hycoclip import lorentz as L
from hycoclip.evaluation.catalog import DatasetCatalog
from hycoclip.evaluation.class_names import CLASS_NAMES
from hycoclip.models import HyCoCLIP, MERU, CLIPBaseline
from hycoclip.tokenizer import Tokenizer
from tqdm import tqdm



from hycoclip.evaluation.coco_eval import CocoEvaluator
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

from torchvision import transforms



parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")


from torchvision import transforms as T
from hycoclip.config import LazyCall as L

# image_transform=[
#     T.RandomResizedCrop(
#         size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
#     ),
#     T.ToTensor(),
# ],  

image_transform = T.Compose([
    T.RandomResizedCrop(
        size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
    ),
    T.ToTensor(),
])

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def main(_A: argparse.Namespace):


    IMG_PATH = "/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/datasets/coco/val2017"
    ANNOTATION_PATH = "/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/datasets/coco/annotations/instances_val2017.json"


    dataset = CocoDetection(root=IMG_PATH, annFile=ANNOTATION_PATH, transform=image_transform)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    evaluator = CocoEvaluator(coco_gt=dataset.coco, iou_types=["bbox"])


    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    logger.info(f"Evaluating checkpoint in {_A.checkpoint_path}...")

    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)


    for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
        # Stack images and move to device
        images = torch.stack(images).to(model.device)
        
        # Encode images
        embeddings = model.encode_image(images, project=True)
        
        # Prepare results in evaluator-friendly format
        results = []
        for i, target in enumerate(targets):
            # Unique annotation ID (can just use a counter)
            ann_id = batch_idx * dataloader.batch_size + i
            # Assuming each target is a dict with 'image_id'
            img_id = target[0]['image_id'] if len(target) > 0 else batch_idx  # fallback to batch_idx
            results.append({
                'id': ann_id,
                'image_id': img_id,
                'embedding': embeddings[i].detach().cpu().numpy()
            })
        
        # Update evaluator
        evaluator.update(results)


    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()



if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
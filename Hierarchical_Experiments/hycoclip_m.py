import json
import torch
import random
import argparse
from PIL import Image
from tqdm import tqdm
import os
import torchvision.transforms as T

from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer
from hycoclip.lorentz import pairwise_inner

def load_hycoclip_model(pretrained, image_size=224, device='cuda'):
    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    model_config = os.path.join(
        os.path.dirname(__file__), "hycoclip_cfg.py"
    )
    cfg = LazyConfig.load(model_config)
    model = LazyFactory.build_model(cfg, device).eval()
    CheckpointManager(model=model).load(pretrained)
    model.to(device).eval()

    tokenizer = Tokenizer()
    preprocess = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )
    return model, preprocess, tokenizer

def get_taxonomy_descriptions_single_style(choices, level_number):
    template = "a photo of a {}."
    return [template.format(choice) for choice in choices]


def infer_level_hycoclip(model, tokenizer, preprocess, image_path, descriptions, choice_map, device):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        text_tokens = tokenizer(descriptions)

        with torch.no_grad(), torch.autocast(device_type=device):
            image_features = model.encode_image(image, project=True)
            text_features = model.encode_text(text_tokens, project=True)

        logits = (pairwise_inner(image_features, text_features)).softmax(dim=-1)
        pred_index = logits.argmax(dim=-1).item()
        predicted_label = list(choice_map.values())[pred_index]
        predicted_letter = list(choice_map.keys())[pred_index]

    except Exception as e:
        print(f"Error in HyCoCLIP inference: {str(e)}")
        predicted_letter = "Unknown"
        predicted_label = "Unknown"

    return predicted_letter, predicted_label


def test_hycoclip(json_file, output_path, model, preprocess, tokenizer, device):
    with open(json_file, "r") as f:
        lines = f.readlines()
        test_data = [json.loads(line) for line in lines]

    results = []
    for entry in tqdm(test_data, desc=f"testing"):
        image_path = entry["image"]
        label = entry["label"]

        level_keys = sorted([k for k in entry.keys() if k.startswith("level") and k[5:].isdigit()], key=lambda x: int(x[5:]))
        choices_keys = sorted([k for k in entry.keys() if k.startswith("choices_level") and k[13:].isdigit()], key=lambda x: int(x[13:]))

        if not level_keys or not choices_keys:
            print(f"Warning: Missing taxonomy data for {image_path}")
            continue

        result_entry = {"image": image_path, "label": label}

        for level_key, choices_key in zip(level_keys, choices_keys):
            level_number = level_key[5:]
            ground_truth = entry[level_key]
            choices = entry[choices_key]

            if ground_truth is None:
                continue

            choice_map = {chr(65 + i): opt for i, opt in enumerate(choices)}
            descriptions = get_taxonomy_descriptions_single_style(choices, level_number)

            predicted_letter, predicted_label = infer_level_hycoclip(
                model, tokenizer, preprocess, image_path, descriptions, choice_map, device
            )

            result_entry[f"ground_truth_level{level_number}"] = ground_truth
            result_entry[f"predicted_level{level_number}_letter"] = predicted_letter
            result_entry[f"predicted_level{level_number}"] = predicted_label
            result_entry[f"choices_level{level_number}"] = choice_map

        results.append(result_entry)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    seed=42
    random.seed(seed)

    parser = argparse.ArgumentParser(description="hycoclip hierarchical classification")
    parser.add_argument("--test_set", type=str,default="/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/Inat_code/Animalia_with_similar_choice.jsonl")
    parser.add_argument("--output_file", type=str, default="/projectnb/ivc-ml/yuwentan/LLaVA-NeXT/CLIP_Style/hycoclip/animal/animal_hierarchy_prompt_hycoclip_new.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="/leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/checkpoints/hycoclip_v2_vit_b.pth")
    args = parser.parse_args()
    model, preprocess, tokenizer = load_hycoclip_model(args.model_path, device=args.device)

    test_hycoclip(args.test_set, args.output_file, model, preprocess, tokenizer, args.device)

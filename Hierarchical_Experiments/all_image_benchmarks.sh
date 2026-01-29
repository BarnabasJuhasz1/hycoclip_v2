#! /bin/bash

source /leonardo/home/userexternal/${USER}/.bashrc

HYPERBOLIC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-dir)
            EXP_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --hyperbolic)
            HYPERBOLIC=1
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$EXP_DIR" ]; then
    echo "Usage: $0 --exp-dir <experiment_directory>"
    return 1
fi

# If MODEL_PATH is set, extract the last two folders and append to $EXP_DIR
if [ -n "$MODEL_PATH" ]; then
    last_two=$(echo "$MODEL_PATH" | awk -F/ '{print $(NF-1) "/" $NF}')
    EXP_DIR="${EXP_DIR}/${last_two}"
fi

# Create the experiment directory if it doesn't exist
mkdir -p "$EXP_DIR"

export HF_HUB_OFFLINE=1


## CLIP
if [ $MODEL == "clip" ]; then   

    # if model path is not set, use the default model path
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH="laion2b_s32b_b82k"
    fi

    conda activate openclip_ft
    python evaluation/clip/openclip.py --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_CUB.json \
    --model_path $MODEL_PATH \
    --hyperbolic $HYPERBOLIC

    python evaluation/clip/openclip.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_inat21_a.json \
    --model_path $MODEL_PATH \
    --hyperbolic $HYPERBOLIC

    python evaluation/clip/openclip.py --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_inat21_p.json \
    --model_path $MODEL_PATH \
    --hyperbolic $HYPERBOLIC

    python evaluation/clip/openclip.py --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_imagenet_art.json \
    --model_path $MODEL_PATH \
    --hyperbolic $HYPERBOLIC

    python evaluation/clip/openclip.py --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_imagenet_a.json \
    --model_path $MODEL_PATH \
    --hyperbolic $HYPERBOLIC

    python evaluation/clip/openclip.py --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_food.json \
    --model_path $MODEL_PATH \
    --hyperbolic $HYPERBOLIC
fi

## SigLIP
if [ $MODEL == "siglip" ]; then
    conda activate hf_vlms
    python evaluation/clip/siglip.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/siglip_inat21_a.json

    python evaluation/clip/siglip.py --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/siglip_inat21_p.json

    python evaluation/clip/siglip.py --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/siglip_imagenet_art.json

    python evaluation/clip/siglip.py --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/siglip_imagenet_a.json

    python evaluation/clip/siglip.py --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/siglip_CUB.json

    python evaluation/clip/siglip.py --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/siglip_food.json
fi

## HyCoCLIP
if [ $MODEL == "hycoclip" ]; then
    conda activate hycoclip
    # python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    # --output_file results/$EXP_DIR/hycoclip_inat21_a.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_inat21_p.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_imagenet_art.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_imagenet_a.json
    
    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_CUB.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_food.json
fi

## HyCoCLIP
if [ $MODEL == "hycoclipV2" ]; then
    conda activate hycoclipFT
    # python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    # --output_file results/$EXP_DIR/hycoclip_inat21_a.json

    python /leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/Hierarchical_Experiments/hycoclip_m.py --test_set /leonardo/home/userexternal/bjuhasz0/fast/projects/amsterdam/Hyperbolic_LLM-Hierarchical-Consistency/data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file $EXP_DIR/hycoclip_inat21_p.json --model_path $MODEL_PATH

    python /leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/Hierarchical_Experiments/hycoclip_m.py --test_set /leonardo/home/userexternal/bjuhasz0/fast/projects/amsterdam/Hyperbolic_LLM-Hierarchical-Consistency/data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file $EXP_DIR/hycoclip_imagenet_art.json --model_path $MODEL_PATH

    python /leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/Hierarchical_Experiments/hycoclip_m.py --test_set /leonardo/home/userexternal/bjuhasz0/fast/projects/amsterdam/Hyperbolic_LLM-Hierarchical-Consistency/data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file $EXP_DIR/hycoclip_imagenet_a.json --model_path $MODEL_PATH
    
    python /leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/Hierarchical_Experiments/hycoclip_m.py --test_set /leonardo/home/userexternal/bjuhasz0/fast/projects/amsterdam/Hyperbolic_LLM-Hierarchical-Consistency/data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file $EXP_DIR/hycoclip_CUB.json --model_path $MODEL_PATH

    python /leonardo/home/userexternal/bjuhasz0/fast/barnabas/project_hierarchies/HyCoCLIP/Hierarchical_Experiments/hycoclip_m.py --test_set /leonardo/home/userexternal/bjuhasz0/fast/projects/amsterdam/Hyperbolic_LLM-Hierarchical-Consistency/data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file $EXP_DIR/hycoclip_food.json --model_path $MODEL_PATH
fi

## HyCoCLIP
if [ $MODEL == "hycoclipv2" ]; then
    conda activate hycoclipFT
    # python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    # --output_file results/$EXP_DIR/hycoclip_inat21_a.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_inat21_p.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_imagenet_art.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_imagenet_a.json
    
    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_CUB.json

    python evaluation/clip/hycoclip_m.py --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hycoclip_food.json
fi

## CLIP from HyCoCLIP
if [ $MODEL == "clip_from_hycoclip" ]; then
    conda activate hycoclip
    # python evaluation/clip/clip_hycoclip.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    # --output_file results/$EXP_DIR/clip_hycoclip_inat21_a.json

    python evaluation/clip/clip_hycoclip.py --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_hycoclip_inat21_p.json

    python evaluation/clip/clip_hycoclip.py --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_hycoclip_imagenet_art.json

    python evaluation/clip/clip_hycoclip.py --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_hycoclip_imagenet_a.json

    python evaluation/clip/clip_hycoclip.py --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_hycoclip_CUB.json

    python evaluation/clip/clip_hycoclip.py --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/clip_hycoclip_food.json
fi

## Hyperbolic CLIP
if [ $MODEL == "hyperbolic_clip" ]; then
    conda activate hf_vlms_ft
    python evaluation/clip/hyperbolic_clip_m.py --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hyperbolic_clip_CUB.json

    python evaluation/clip/hyperbolic_clip_m.py --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hyperbolic_clip_imagenet_art.json

    python evaluation/clip/hyperbolic_clip_m.py --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hyperbolic_clip_imagenet_a.json

    python evaluation/clip/hyperbolic_clip_m.py --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hyperbolic_clip_inat21_p.json

    python evaluation/clip/hyperbolic_clip_m.py --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hyperbolic_clip_inat21_a.json

    python evaluation/clip/hyperbolic_clip_m.py --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/hyperbolic_clip_food.json
fi

## LLaVA-OV
if [ $MODEL == "llava" ]; then
    conda activate llava
    python evaluation/vllm/llava/eval_natural_animal.py --prompt_order 0  \
    --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/llava_inat21_a.json

    python evaluation/vllm/llava/eval_natural_plant.py --prompt_order 0  \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/llava_inat21_p.json

    python evaluation/vllm/llava/eval_artifact.py --prompt_order 0  \
    --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/llava_imagenet_art.json

    python evaluation/vllm/llava/eval_animal.py --prompt_order 0  \
    --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/llava_imagenet_a.json

    python evaluation/vllm/llava/eval_CUB.py --prompt_order 0  \
    --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/llava_CUB.json

    python evaluation/vllm/llava/eval_food.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/llava_food.json
fi

## InternVL2.5-8B
if [ $MODEL == "internvl2.5" ]; then
    conda activate internvl
    python evaluation/vllm/internvl/eval_img_natrual_ani.py --prompt_order 1 \
    --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl2-5-8B_inat21_a.json

    python evaluation/vllm/internvl/eval_img_natrual_plant.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl2-5-8B_inat21_p.json

    python evaluation/vllm/internvl/eval_img_artifact.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl2-5-8B_imagenet_art.json

    python evaluation/vllm/internvl/eval_img_animal.py --prompt_order 2\
    --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl2-5-8B_imagenet_a.json

    python evaluation/vllm/internvl/eval_img_cub.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl2-5-8B_CUB.json

    python evaluation/vllm/internvl/eval_img_food.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl2-5-8B_food.json
fi

## InternVL3-8B
if [ $MODEL == "internvl3" ]; then
    conda activate internvl
    python evaluation/vllm/internvl/eval_img_natrual_ani.py --prompt_order 1 \
    --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl3-8B_inat21_a.json \
    --internvl_3

    python evaluation/vllm/internvl/eval_img_natrual_plant.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl3-8B_inat21_p.json \
    --internvl_3

    python evaluation/vllm/internvl/eval_img_artifact.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl3-8B_imagenet_art.json \
    --internvl_3

    python evaluation/vllm/internvl/eval_img_animal.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl3-8B_inat21_a.json \
    --internvl_3

    python evaluation/vllm/internvl/eval_img_cub.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl3-8B_inat21_p.json \
    --internvl_3

    python evaluation/vllm/internvl/eval_img_food.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --output_file results/$EXP_DIR/internvl3-8B_food.json \
    --internvl_3
fi

## Qwen2.5-VL-7B-Instruct
if [ $MODEL == "qwen7b" ]; then

    # if model path is not set, use the default model path
    if [ -z "$MODEL_PATH" ]; then
        MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
    fi

    conda activate hf_vlms
    python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --model_path $MODEL_PATH \
    --output_file results/$EXP_DIR/qwen2-5-7B_CUB.json
    
    python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
    --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --model_path $MODEL_PATH \
    --output_file results/$EXP_DIR/qwen2-5-7B_inat21_a.json

    python evaluation/vllm/qwen/eval_animal.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --model_path $MODEL_PATH \
    --output_file results/$EXP_DIR/qwen2-5-7B_imagenet_a.json

    python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 1 \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --model_path $MODEL_PATH \
    --output_file results/$EXP_DIR/qwen2-5-7B_inat21_p.json

    python evaluation/vllm/qwen/eval_artifact.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --model_path $MODEL_PATH \
    --output_file results/$EXP_DIR/qwen2-5-7B_imagenet_art.json

    python evaluation/vllm/qwen/eval_food.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --model_path $MODEL_PATH \
    --output_file results/$EXP_DIR/qwen2-5-7B_food.json
fi

## Qwen2.5-VL-32B-Instruct
if [ $MODEL == "qwen32b" ]; then
    conda activate hf_vlms
    python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
    --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-32B_inat21_a.json

    python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-32B_inat21_p.json

    python evaluation/vllm/qwen/eval_artifact.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-32B_imagenet_art.json

    python evaluation/vllm/qwen/eval_animal.py --prompt_order  0\
    --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-32B_imagenet_a.json

    python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-32B_CUB.json

    python evaluation/vllm/qwen/eval_Food101.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-32B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-32B_food.json
fi

## Qwen2.5-VL-72B-Instruct
if [ $MODEL == "qwen72b" ]; then
    conda activate hf_vlms
    python evaluation/vllm/qwen/eval_natural_animal.py --prompt_order 1 \
    --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-72B_inat21_a.json

    python evaluation/vllm/qwen/eval_natural_plant.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-72B_inat21_p.json

    python evaluation/vllm/qwen/eval_artifact.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-72B_imagenet_art.json

    python evaluation/vllm/qwen/eval_animal.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/imagenet_animal_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-72B_imagenet_a.json

    python evaluation/vllm/qwen/eval_CUB.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-72B_CUB.json

    python evaluation/vllm/qwen/eval_Food101.py --prompt_order 0 \
    --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
    --model_path Qwen/Qwen2.5-VL-72B-Instruct \
    --output_file results/$EXP_DIR/qwen2-5-72B_food.json
fi

# ## GPT-4O

# python evaluation/vllm/gpt/gpt_img_all.py  \
# --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
# --output_file results/$EXP_DIR/gpt_inat21_a.json \
# --dataset_prompt natural_animal

# python evaluation/vllm/gpt/gpt_img_all.py \
# --test_set data/annotations/similar_choices/inat21_plantae_with_similar_choice.jsonl \
# --output_file results/$EXP_DIR/gpt_inat21_p.json \
# --dataset_prompt natural_plant

# python evaluation/vllm/gpt/gpt_img_all.py \
# --test_set data/annotations/similar_choices/imagenet_artifact_with_similar_choice.jsonl \
# --output_file results/$EXP_DIR/gpt_imagenet_art.json \
# --dataset_prompt artifact

# python evaluation/vllm/gpt/gpt_img_all.py \
# --test_set data/annotations/similar_choices/inat21_animalia_with_similar_choice.jsonl \
# --output_file results/$EXP_DIR/gpt_inat21_a.json \
# --dataset_prompt animal

# python evaluation/vllm/gpt/gpt_img_all.py \
# --test_set data/annotations/similar_choices/CUB200_with_similar_choice.jsonl \
# --output_file results/$EXP_DIR/gpt_CUB.json \
# --dataset_prompt cub

# python evaluation/vllm/gpt/gpt_img_all.py \
# --test_set data/annotations/similar_choices/Food101_with_similar_choice.jsonl \
# --output_file results/$EXP_DIR/gpt_food.json \
# --dataset_prompt food

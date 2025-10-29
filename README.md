# CopyJudge: Automated Copyright Infringement Identification and Mitigation in Text-to-Image Diffusion Models

[![Paper](https://img.shields.io/badge/Paper-ACM_MM_2025-blue )](https://arxiv.org/abs/2502.15278)
[![License](https://img.shields.io/badge/License-MIT-green.svg )](LICENSE)

## Overview
This repository contains two Python scripts for the identification and mitigation of copyright infringement in text-to-image models:

1. **copyright_detection.py**: Code for detecting copyright infringement across different datasets
2. **copyright_attack_defense.py**: Code for automated attack and defense against copyright infringement

Both scripts require an OpenAI API key for proper functionality. The following instructions guide you through setting up and using the code.

## Requirements

### Python Libraries
Before running the scripts, install the following Python packages:

**For copyright_detection.py:**
```bash
pip install openai
```

**For copyright_attack_defense.py:**
```bash
pip install torch openai diffusers
```

### Standard Python Libraries
The scripts also use these standard libraries (pre-installed with Python):
- os
- re
- csv
- base64
- copy
- json
- math
- argparse
- random (for copyright_detection.py)
- collections (for copyright_detection.py)

## Dataset

### D-Rep Dataset
Download the prior demonstration dataset from [Hugging Face: D-Rep Dataset](https://huggingface.co/datasets/WenhaoWang/D-Rep).
- **Training set**: Used for prior demonstrations (few-shot learning)
- **Test set**: Used for testing infringement detection

### Cartoon and Artwork Dataset
Download our custom dataset from [Google Drive](https://drive.google.com/file/d/1Fyr3l0vfErSw3YSD9HvPEc-oeDgdW6ha/view?usp=sharing).

**Note**: The current dataset is relatively small. We are planning to expand it with more test cases. If you are interested in collaborating or obtaining the expanded dataset, please contact us at: **shunchangliu@outlook.com**

## Setup

1. **Register for OpenAI API Key:**
   - Follow the instructions on [OpenAI's website](https://platform.openai.com/) to obtain your API key
   - You will need to provide this key as a command-line argument when running the scripts

2. **Download Datasets:**
   - Download the D-Rep dataset from Hugging Face for detection tasks
   - Download the cartoon/artwork dataset from Google Drive for attack/defense tasks

3. **Prepare Directory Structure:**
   - Organize your datasets according to the expected structure (see Usage section below)

## Usage

### 1. Infringement Detection (copyright_detection.py)

This script detects copyright infringement by comparing image pairs and assigning similarity scores.

#### Supported Datasets
- **D-Rep**: Paired images with human-labeled similarity scores
- **Cartoon**: Cartoon character images with synthetic variations
- **Artwork**: Artwork images with synthetic variations

#### Basic Usage

**For D-Rep Dataset:**
```bash
python copyright_detection.py \
    --dataset_type drep \
    --api_key YOUR_OPENAI_API_KEY \
    --demo_path /path/to/training_pairs \
    --labels_path /path/to/labels.csv \
    --test_path /path/to/test_pairs \
    --output_file detection_results_drep.json
```

**For Cartoon Dataset:**
```bash
python copyright_detection.py \
    --dataset_type cartoon \
    --api_key YOUR_OPENAI_API_KEY \
    --demo_path /path/to/training_pairs \
    --labels_path /path/to/labels.csv \
    --raw_dir /path/to/cartoon/raw \
    --artwork_base /path/to/cartoon \
    --output_file detection_results_cartoon.json
```

**For Artwork Dataset:**
```bash
python copyright_detection.py \
    --dataset_type artwork \
    --api_key YOUR_OPENAI_API_KEY \
    --demo_path /path/to/training_pairs \
    --labels_path /path/to/labels.csv \
    --raw_dir /path/to/artwork/raw \
    --artwork_base /path/to/artwork \
    --output_file detection_results_artwork.json
```

#### Key Parameters
- `--dataset_type`: Dataset type (required: drep/cartoon/artwork)
- `--api_key`: OpenAI API key (required)
- `--demo_path`: Path to demonstration/training pairs directory (required)
- `--labels_path`: Path to labels CSV file (required)
- `--test_path`: Path to test pairs directory (required for D-Rep)
- `--raw_dir`: Path to raw images directory (required for cartoon/artwork)
- `--artwork_base`: Base path for cartoon/artwork directory (required for cartoon/artwork)
- `--output_file`: Output JSON file path (required)
- `--convergence_threshold`: Threshold for debate convergence (default: 0.1)
- `--num_agents`: Number of judging agents (default: 3)
- `--max_rounds`: Maximum number of debate rounds (default: 5)


### 2. Infringement Attack and Mitigation (copyright_attack_defense.py)

This script performs automated attacks to generate infringing content and optionally applies mitigation strategies to reduce similarity.

#### Supported Datasets
- **Cartoon**: Cartoon character images
- **Artwork**: Artwork images

#### Attack Types
- **Explicit Attack**: Generated prompts include the IP name (e.g., "Generate a cartoon like Mickey Mouse...")
- **Implicit Attack**: Generated prompts do not include the IP name (e.g., "Generate a cartoon...")

#### Mitigation Strategies
- **PC (Prompt Correction)**: Modifies the text prompt to reduce similarity
- **LC (Latent Correction)**: Adjusts the latent space representation
- **LCPC (Combined)**: Uses both prompt and latent correction

#### Basic Usage

**Attack + Defense (Combined LCPC):**
```bash
python copyright_attack_defense.py \
    --dataset_type cartoon \
    --api_key YOUR_OPENAI_API_KEY \
    --test_path /path/to/cartoon/raw \
    --output_path /path/to/output \
    --attack_type implicit \
    --attack_iter 5 \
    --attack_threshold 0.8 \
    --run_defense \
    --use_prompt_correction \
    --use_latent_correction \
    --mitigation_iter 10 \
    --mitigation_threshold 0.6
```

#### Key Parameters

**Common Parameters:**
- `--dataset_type`: Dataset type (required: cartoon/artwork)
- `--api_key`: OpenAI API key (required)
- `--test_path`: Path to test images directory (required)
- `--output_path`: Path to output directory (required)
- `--model_id`: Diffusion model ID (default: 'stabilityai/stable-diffusion-2')

**Attack Parameters:**
- `--attack_type`: Attack type (choices: explicit/implicit, default: explicit)
- `--attack_iter`: Number of attack iterations (default: 5)
- `--attack_threshold`: Score threshold to stop attack when >= threshold (default: 0.8)

**Defense Parameters:**
- `--run_defense`: Enable defense phase after attack (flag)
- `--use_prompt_correction`: Enable prompt correction (flag)
- `--use_latent_correction`: Enable latent correction (flag)
- `--mitigation_iter`: Number of mitigation iterations (default: 1)
- `--mitigation_threshold`: Score threshold to stop mitigation when <= threshold (default: 0.6)

**Optional Parameters:**
- `--demo_path`: Path to demonstration directory for few-shot learning
- `--labels_path`: Path to labels CSV file (required if demo_path provided)
- `--num_demonstrations`: Number of demonstrations per label score (default: 3)
- `--convergence_threshold`: Threshold for debate convergence (default: 0.1)
- `--num_agents`: Number of judging agents (default: 3)
- `--max_rounds`: Maximum number of debate rounds (default: 5)

#### Output
The script generates:
- **Generated images**: Saved in the output directory
- **Attack results**: JSON file with format `adv_{attack_type}_{dataset}.json`
- **Defense results**: JSON file with format `{method}_{dataset}.json` where method is pc/lc/lcpc/none

Each JSON entry contains:
- IP name or image identifier
- Generated prompts
- Similarity scores
- Confidence levels
- Reasoning/rationale

## Hardware Requirements

- **copyright_detection.py**: Can run on CPU, but GPU recommended for faster processing
- **copyright_attack_defense.py**: Requires CUDA-capable GPU for running diffusion models



## 📚 Citation

If you find our work useful, please cite:

```bibtex
@article{liu2025copyjudge,
  title={CopyJudge: Automated Copyright Infringement Identification and Mitigation in Text-to-Image Diffusion Models},
  author={Liu, Shunchang and Shi, Zhuan and Lyu, Lingjuan and Jin, Yaochu and Faltings, Boi},
  journal={arXiv preprint arXiv:2502.15278},
  year={2025}
}
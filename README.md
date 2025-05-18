# READ-CLIP 🏹📚

**REconstruction and Alignment of text Descriptions for Compositional Reasoning in CLIP**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](add_paper_link_here)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](add_demo_link_here)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-lightgrey?logo=pytorch)](https://pytorch.org)

READ-CLIP is a lightweight fine-tuning recipe that plugs a **frozen text decoder** into CLIP and adds two auxiliary
losses—**token-level reconstruction** and **sentence-level alignment**—to unlock state-of-the-art compositional
reasoning.  
Trained on only **100 k MS-COCO samples**, READ-CLIP (ViT-B/32) tops five standard benchmarks, beating strong baselines
such as NegCLIP and FSC-CLIP by up to **4.5 pp**.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Training](#training)
4. [Evaluation & Results](#evaluation--results)
5. [Pre-trained Checkpoints](#pre-trained-checkpoints)
6. [Reproducing the Paper](#reproducing-the-paper)
7. [Expected Compute & Determinism](#expected-compute--determinism)
8. [License](#license)

---

## Quick Start

```bash
# 1. create env (Python ≥3.8, PyTorch ≥2.1)
conda create -n readclip python=3.10 -y && conda activate readclip
pip install -r requirements.txt

# 3. run zero-shot inference
python inference.py   --model-path checkpoints/read-clip_vitb32.pt   --image-path examples/cat_monitor.jpg
```

---

## Installation

<details>
<summary><strong>Option A · Pip / Conda (recommended)</strong></summary>

```bash
bash setup.sh         # installs package in editable mode
```

</details>

<details>
<summary><strong>Option B · Docker</strong></summary>

```bash
docker build -t read-clip .
docker run --gpus all -it read-clip
```

</details>

<details>
<summary><strong>Option C · Docker Compose (one-liner)</strong></summary>

```bash
docker compose up -d         # start
docker compose exec read-clip bash
docker compose down          # stop & remove
```

</details>

---

## Training

```bash
python train.py   --cfg-path config/train_read_clip.yaml   --wandb-key <YOUR_WANDB_KEY>        # optional
```

Key hyper-parameters (defined in the YAML):

| name               | value  | note   |
|--------------------|--------|--------|
| `learning_rate`    | `1e-5` | AdamW  |
| `weight_decay`     | `1e-1` |        |
| `num_train_epochs` | `5`    |        |
| `batch_size`       | `256`  | global |
| `bf16`             | `true` | A100   |

---

## Evaluation & Results

```bash
python evaluate.py --cfg-path config/eval_read_clip.yaml
```

| Benchmark          | Metric | READ-CLIP | NegCLIP | FSC-CLIP |
|--------------------|--------|-----------|---------|----------|
| WhatsUp            | Acc.   | **43.9**  | 42.4    | 39.8     |
| VALSE              | Acc.   | **76.2**  | 73.7    | 74.4     |
| CREPE              | Acc.   | **41.5**  | 30.5    | 42.5     |
| SugarCrepe         | Acc.   | **87.0**  | 83.6    | 85.2     |
| SugarCrepe++ (ITT) | Acc.   | **69.8**  | 65.0    | 67.9     |
| SugarCrepe++ (TOT) | Acc.   | **66.2**  | 62.5    | 64.4     |
| **Average**        | Acc.   | **64.1**  | 59.6    | 62.4     |

Numbers reproduce Table 1 in the paper.

---

## Pre-trained Checkpoints

| model              | link                                                    |
|--------------------|---------------------------------------------------------|
| READ-CLIP ViT-B/32 | [Checkpoint](https://huggingface.co/Mayfull/READ-CLIP). |

---

## Reproducing the Paper

```bash
bash scripts/eval.sh
```

Full scripts for every table and figure live in **`scripts/`**.

---

## Expected Compute & Determinism

* All results are obtained on **one NVIDIA A100 40 GB** GPU.
* Training READ-CLIP ViT-B/32 (5 epochs, 256 batch) takes **≈2 GPU‑hours**.
* We fix `torch`, `numpy`, and `random` seeds to `2025` for determinism.

---

## License

Released under the **MIT License**—see [`LICENSE`](LICENSE).

---

<sub>Last updated · 2025‑05‑18</sub>

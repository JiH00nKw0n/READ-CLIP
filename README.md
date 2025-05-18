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

1. [Quick Start](#quick-start-sample-inference)
2. [Installation & Usage (Docker Recommended)](#installation--usage-docker-recommended)
3. [Training](#advanced-entering-a-shell-inside-docker)
4. [Experiments & Results](#experiments--results)
5. [Reproducing the Paper](#reproducing-the-paper)
6. [Pre-trained Checkpoints](#pre-trained-checkpoints)
7. [Expected Compute & Determinism](#expected-compute--determinism)
8. [License](#license)

---

## Quick Start [Sample Inference]

```bash
# 1. create env (Python ≥3.8, PyTorch ≥2.1)
conda create -n readclip python=3.10 -y && conda activate readclip
pip install -r requirements.txt

# 3. run zero-shot inference
python example.py
```

---

## Installation & Usage (Docker Recommended)

Everything can be run in Docker — no Python installation or CUDA drivers on the host required (except for the NVIDIA
driver).

<details>
<summary><strong>Step 1. Build the Docker image</strong></summary>

```bash
docker build -t read-clip .
```

</details>

<details>
<summary><strong>Step 2. Training</strong></summary>

```bash
bash run_docker.sh --train --wandb-key YOUR-WANDB-KEY
```

- All necessary data, output, and logs directories will be mounted for persistence.
- `YOUR-WANDB-KEY` is optional; if omitted, W&B logging will be disabled.

</details>

<details>
<summary><strong>Step 3. Evaluation</strong></summary>

```bash
bash run_docker.sh --eval
```

</details>

---

### Advanced: Entering a Shell Inside Docker

If you want to run custom scripts or debug:

```bash
bash run_docker.sh
```

Then, inside the container:

```bash
source /venv/bin/activate
bash setup.sh
# Now you can run anything, e.g.
python train.py --cfg-path config/train_read_clip.yaml
```

---

> **Tip:**  
> For convenient mode switching (`train`/`eval`/`shell`), use the provided [run_docker.sh](./run_docker.sh) launcher:
> ```bash
> bash run_docker.sh --train --wandb-key YOUR-WANDB-KEY      # for training
> bash run_docker.sh --eval                                  # for evaluation
> bash run_docker.sh                                         # just get a shell
> ```

---

## Experiments & Results

Key hyper-parameters (defined in the YAML):

| name               | value  | note   |
|--------------------|--------|--------|
| `learning_rate`    | `1e-5` | AdamW  |
| `weight_decay`     | `1e-1` |        |
| `num_train_epochs` | `5`    |        |
| `batch_size`       | `256`  | global |
| `bf16`             | `true` | A100   |


## Reproducing the Paper

> ```bash
> bash run_docker.sh --train --wandb-key YOUR-WANDB-KEY      # for training
> ```

```bash
> bash run_docker.sh --eval
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

## Expected Compute & Determinism

* All results are obtained on **one NVIDIA A100 40 GB** GPU.
* Training READ-CLIP ViT-B/32 (5 epochs, 256 batch) takes **≈2 GPU‑hours**.
* We fix `torch`, `numpy`, and `random` seeds to `2025` for determinism.

---

## License

Released under the **MIT License**—see [`LICENSE`](LICENSE).

---

<sub>Last updated · 2025‑05‑18</sub>

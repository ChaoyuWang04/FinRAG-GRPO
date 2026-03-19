# FinRAG-GRPO

<div align="center">
  <h3 align="center">Reasoning Reward Model Training Workflow for Customer Service Preference Data</h3>
  <p align="center">
    This repository contains a practical RM-R1 style pipeline: generate pairwise customer-service preference data, preprocess and split it, train a reasoning reward model with GRPO on top of veRL/vLLM/Ray, and run inference with an exported checkpoint.
    <br /><br />
    <a href="https://github.com/ChaoyuWang04/FinRAG-GRPO">Repository</a> |
  </p>
</div>

## About The Project

This repository implements a **Reasoning Reward Model (ReasRM)** workflow. Instead of directly predicting a scalar reward, the model is trained to behave like a reviewer: it reads a customer question plus two candidate answers, reasons about which answer is better, and finally emits a structured decision such as `<answer>[[A]]</answer>` or `<answer>[[B]]</answer>`.

The current project is centered on a Chinese e-commerce customer-service use case. The training data compares a weaker, more mechanical answer with a better, more empathetic and solution-oriented answer. The training recipe skips a separate SFT stage and directly applies **GRPO / RLVR-style reinforcement learning** on a reasoning-capable base model.

At a high level, the workflow in this repository is:

1. Generate synthetic pairwise preference samples.
2. Merge, shuffle, and split them into train/test JSONL files.
3. Optionally inject a system prompt into every example.
4. Launch GRPO training with veRL, Ray, and vLLM.
5. Export the trained checkpoint and run inference with Transformers.

### Built With

- Python 3.11
- [veRL](https://github.com/volcengine/verl)
- [vLLM](https://github.com/vllm-project/vllm)
- [Ray](https://github.com/ray-project/ray)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)

## Getting Started

The repository does not currently ship a single `requirements.txt` or one-command bootstrap script. The expected setup is the same one described in [README_zh.md](./README_zh.md): create a Python environment, install pinned veRL and vLLM revisions, then run the local scripts after adapting paths to your machine.

### Prerequisites

- Linux environment with NVIDIA GPUs
- Python 3.11
- Conda or another environment manager
- CUDA-compatible PyTorch environment
- A local base model checkpoint for training
- veRL and vLLM installed from source

### Environment Setup

```sh
conda create -n rm-r1 python=3.11 -y
conda activate rm-r1
```

Install veRL at the revision referenced by the Chinese README:

```sh
git clone https://github.com/volcengine/verl
cd verl
git checkout e49fb572bf85a8f0ef7124c898f509bd6d9832a1
pip install -e .
```

Install vLLM at the pinned revision used by this workflow:

```sh
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout ed6e9075d31e32c8548b480a47d1ffb77da1f54c
git cherry-pick caac5c2e597b1780c3df54a537c34e6061c32cff
export VLLM_COMMIT=ed6e9075d31e32c8548b480a47d1ffb77da1f54c
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install --editable .
pip install flash-attn==2.7.2.post1 --no-build-isolation
```

You will also need the common runtime libraries used by the helper scripts, for example:

```sh
pip install torch transformers tqdm
```

### Repository Structure

```text
FinRAG-GRPO/
├── README.md
├── README_zh.md
├── docs/note.md                               # Chinese technical walkthrough of the full pipeline
├── generate_customer_service_data.py          # Synthetic pairwise data generation
├── merge_and_split_dataset.py                 # Merge, shuffle, and split JSONL datasets
├── demo/demo.py                               # Inference demo with an exported HF checkpoint
├── demo/convert_fsdp_to_hf.py                 # Convert training output into HF-style weights
└── rm_r1/
    ├── dataset/mix_data/                      # Raw, merged, and system-prompted datasets
    ├── scripts/RLVR/local/                    # Local GRPO training shell scripts
    └── verl/                                  # Customized trainer, dataset, reward, and worker code
```

## Usage

### 1. Generate synthetic preference data

`generate_customer_service_data.py` creates JSONL records in the format expected by the RM dataset loader:

- `context_messages`: a chat-style prompt containing the customer question and answers A/B
- `winner`: `model_a` or `model_b`

The script is designed for Chinese customer-service scenarios such as logistics delays, refunds, price disputes, address changes, and payment issues.

Run it after providing your own `llm.call_llm(...)` implementation or equivalent local wrapper:

```sh
python generate_customer_service_data.py
```

By default it:

- targets 3000 synthetic samples
- uses multithreaded generation
- writes `customer_service_dataset.jsonl`
- randomizes A/B order to reduce position bias

### 2. Merge and split datasets

`merge_and_split_dataset.py` combines multiple JSONL files, shuffles them with a fixed seed, and writes train/test splits:

```sh
python merge_and_split_dataset.py
```

Default outputs:

- `rm_r1/dataset/mix_data/train.jsonl`
- `rm_r1/dataset/mix_data/test.jsonl`

### 3. Inject the system prompt

If you want every training example to begin with the Chinese rubric/system prompt, run:

```sh
cd rm_r1/dataset/mix_data
python preprocess_data.py
```

This creates:

- `train_with_sys.jsonl`
- `test_with_sys.jsonl`

### 4. Launch GRPO training

The main local training entrypoint is:

```sh
bash ./rm_r1/scripts/RLVR/local/train_rm_r1_rlvr_dpsk_distilled_7b.sh
```

The training script:

- configures Ray and GPU usage
- points veRL to the train/validation JSONL files
- loads a base reasoning model
- uses `rm_r1/verl/utils/reward_score/lm_as_judge.py` as the reward function
- starts PPO/GRPO-style optimization through `rm_r1.verl.trainer.main_ppo`

Important: the script currently contains **machine-specific absolute paths** such as `/root/autodl-tmp/...`. You should update at least the following before running it:

- `MODEL_PATH`
- `SAVE_META_DIR`
- `TRAIN_TASK`
- `EVAL_TASK`

### 5. Export and run inference

After training, the repository includes a simple conversion and demo path:

```sh
python demo/convert_fsdp_to_hf.py
python demo/demo.py
```

`demo/demo.py` loads the merged model with Transformers, formats a single A/B evaluation prompt, and prints the model's final judgment.

## How The Training Loop Works

The main logic in this repository is:

1. `rm_r1/verl/utils/dataset/rl_dataset.py` reads JSONL preference samples.
2. The prompt is formatted with a chat template and sent to the policy model.
3. The model generates reasoning text plus a final `<answer>[[A/B]]</answer>` decision.
4. `rm_r1/verl/utils/reward_score/lm_as_judge.py` checks whether the predicted final tag matches the ground-truth winner.
5. GRPO/PPO updates are applied through the customized veRL trainer code under `rm_r1/verl/`.

This makes the project a compact prototype for training a reasoning-style reward model rather than a standard scalar reward head.

## Current Caveats

- Several scripts are tightly coupled to one local environment and need path cleanup before reuse.
- `generate_customer_service_data.py` depends on `from llm import call_llm`, but that helper is not included in this repository.
- The training shell script currently sets `EVAL_TASK` to the training set path by default.
- `demo/convert_fsdp_to_hf.py` uses a simplified merge approach and explicitly notes that it is not a full FSDP state-dict gather implementation.
- There is no unified dependency lockfile or end-to-end reproducible bootstrap script yet.

## Roadmap

- [x] Synthetic pairwise customer-service preference data generation
- [x] JSONL merge/split preprocessing workflow
- [x] Optional system-prompt injection for rubric-style training
- [x] GRPO training entrypoint based on veRL, Ray, and vLLM
- [x] Checkpoint export and inference demo
- [ ] Replace machine-specific paths with configurable project-relative paths
- [ ] Add a proper dependency file and reproducible environment setup
- [ ] Improve reward shaping beyond final-tag matching
- [ ] Add a clearer evaluation pipeline and model card outputs

## Contributing

Contributions are welcome, especially in the following areas:

- environment reproducibility
- training/evaluation cleanup
- reward function improvements
- dataset governance and quality checks
- documentation

Typical flow:

1. Fork the repository.
2. Create a branch.
3. Make your changes.
4. Open a pull request with a clear description of the change and how it was validated.

## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for details.

## Contact

Chaoyu Wang  
[LinkedIn](https://www.linkedin.com/in/samwang04/)  
[Personal Website](https://chaoyuwang04.github.io/)

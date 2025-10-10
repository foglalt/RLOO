# RLOO Code Completion Project

This repository contains a minimal reinforcement learning setup that applies **Reinforcement Learning with Online Optimization (RLOO)** to fine-tune a small language model for code-completion tasks. The project is tailored for execution on a Kaggle Jupyter GPU notebook, but it can be adapted to other environments with minor changes.

## Project Highlights

- ✅ Uses [TRL](https://github.com/huggingface/trl) implementations of RLOO on top of Hugging Face Transformers.
- ✅ Includes a lightweight Python package with data, model, and training utilities.
- ✅ Provides ready-to-run Kaggle notebook workflow for dataset preview and training orchestration.
- ✅ Keeps resource requirements modest by defaulting to compact models and tiny datasets.

## Repository Layout

```
.
├─ configs/              # YAML configuration files for experiments
│  └─ default_training.yaml
├─ data/
│  ├─ raw/               # (Optional) place for downloaded raw data
│  └─ processed/         # Cached/processed datasets
├─ notebooks/
│  ├─ 01_dataset_preview.ipynb
│  └─ 02_rloo_training.ipynb
├─ scripts/
│  └─ kaggle_setup.py    # Convenience script for Kaggle notebook environment tweaks
├─ src/
│  └─ rloo_project/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py
│     ├─ modeling.py
│     ├─ prompts.py
│     └─ train_rloo.py
├─ requirements.txt
└─ pyproject.toml
```

## Quickstart on Kaggle

1. **Create a new Kaggle notebook** (GPU preferred). Choose the "Internet" option if you need to download datasets/models that are not cached in Kaggle yet.
2. **Add this repository** as a dataset or upload it via the Kaggle web UI (`Add data > Upload`).
3. **Install dependencies** inside the notebook cell:
   ```python
   %pip install -r /kaggle/input/rloo/requirements.txt
   ```
   Adjust the path if you renamed the dataset.
4. **Wire the project** by adding the repository to `sys.path`:
   ```python
   import sys
   sys.path.append("/kaggle/input/rloo/src")
   ```
5. **Open and run the notebook** `notebooks/02_rloo_training.ipynb` to launch training, or run the Python entry point:
   ```python
   !python /kaggle/input/rloo/src/rloo_project/train_rloo.py --config /kaggle/input/rloo/configs/default_training.yaml
   ```

## Customizing the Experiment

- Edit the YAML configuration (`configs/default_training.yaml`) to change hyperparameters, datasets, or output directories.
- Modify `src/rloo_project/prompts.py` to adjust prompt templates that define the code-completion tasks.
- Extend `src/rloo_project/data.py` if you want to plug in a different dataset or custom sampler.

## Dataset Suggestions

For quick experimentation, consider:
- [`codeparrot/codeparrot-clean`](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
- [`bigcode/the-stack-smol`](https://huggingface.co/datasets/bigcode/the-stack-smol)

Each dataset can be filtered to a handful of examples to keep training light enough for Kaggle GPU constraints.

## Logging & Checkpoints

By default, checkpoints and logs are saved under `/kaggle/working/rloo_runs`. Adjust this path in the configuration file if you prefer a different location.

## Next Steps

- Build richer reward models or heuristics to score completions instead of the default high-level metric.
- Plug in a larger base model if GPU memory allows.
- Integrate experiment tracking (Weights & Biases, MLflow) by extending the training script hooks.

Happy experimenting!

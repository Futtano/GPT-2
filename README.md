# GPT-2 from scratch

## TL;DR

This repo was created with two main goals:

- Understand how the vanilla GPT decoder-style LLM architecture works under the hood by implementing it from scratch in PyTorch.
- Learn how to transform Jupyter notebook experiments into a reproducible, production-grade Python package for deep learning model pre-training and fine-tuning.

## Purpose

This project currently contains a from-scratch GPT-2 implementation and the notebooks used to develop it. Its intended behavior is to provide reproducible command-line workflows for pre-training and fine-tuning small GPT-2 models from local text files.

It is a learning project for people who want to explore LLM training and fine-tuning without access to expensive GPU clusters or large datasets.

The [engineering notes](docs/engineering-notes.md) collect the packaging,
testing, and ML workflow lessons learned while developing the project.

## First supported workflow (TODO)

Start a pre-training job through a CLI using a local text file and a configuration file. A small set of CLI options may override values such as the device and output directory.

The text is tokenized and divided into disjoint training, validation, and test sections before overlapping token windows are created. This prevents nearly identical windows from appearing in more than one split. PyTorch data loaders then provide batches for training and evaluation.

Validation loss selects the best model checkpoint. Separate training checkpoints are saved periodically and contain the model state, optimizer state, and training progress required to resume a run. After training, the selected model is evaluated once on the test split. Loss and perplexity are recorded for each evaluated split.

### Inputs

- A path to a local text dataset, supplied as a CLI argument. The path may be outside the repository. (TODO)
- A configuration file containing model settings and training settings such as batch size, learning rate, training duration, checkpoint frequency, random seed, and device. (TODO)
- A user-selected output directory, which may be outside the repository. (TODO)
- Optional CLI overrides for a small set of operational settings, including the device and output directory. (TODO)

### Outputs

Each run stores the following artifacts in its selected output directory:

- The best model checkpoint, selected by validation loss. (TODO)
- Periodic training checkpoints that can resume training. (TODO)
- The resolved model and training configuration, including defaults and CLI overrides. (TODO)
- Dataset metadata identifying the source data and the offsets used for each split. (TODO)
- A tabular metrics history containing training and validation loss and perplexity, plus final test metrics. (TODO)
- Training and validation loss curves. (TODO)

### Acceptance criteria

- A tiny configuration completes a short training run on CPU.
- The run saves its resolved configuration, dataset split metadata, metrics, and checkpoints in the chosen output directory.
- The best model checkpoint can be loaded and used for text generation.
- Training can resume from a training checkpoint with its optimizer state and step count restored.
- The final test metrics are calculated without using the test split for training or checkpoint selection.

# Project roadmap

This document tracks the current implementation state and the next concrete
milestones for turning the educational GPT-2 code into a reproducible training
package. Update it when a milestone is completed, reordered, or removed.

Durable lessons and technical explanations belong in the
[engineering notes](engineering-notes.md).

## Product target

A user can start a tiny CPU pre-training run from the installed CLI using a
local text file, a TOML configuration, and a new output directory.

A completed run must:

1. create reproducible train, validation, and test splits;
2. record resolved configuration and dataset metadata;
3. record training, validation, and final test metrics;
4. save the best model selected by validation loss;
5. save resumable model, optimizer, and progress checkpoints;
6. load the selected model for text generation;
7. resume training without losing optimizer or progress state.

## Current stage: assemble the pre-training application boundary

Completed in this stage:

- Added a strict TOML loader that constructs a frozen RunConfig.
- Added a checked-in tiny CPU configuration example.
- Added typed parsing for configuration, input, and output paths.
- Added run preparation that validates input files, loads configuration, and
  creates a fresh output directory without leaving partial artifacts on
  failure.
- Added hermetic unit tests for the parser and preparation boundary.

Currently implementing:

- Resolve explicit and automatic CPU, CUDA, and MPS device selection.
- Seed every random source used by the training workflow.
- Attach resolved runtime state to the prepared run.

## Next milestones

### 1. Build the executable training workflow

- Read and tokenize the local text input.
- Construct disjoint token splits and reproducible data loaders.
- Instantiate the model and optimizer from RunConfig.
- Connect evaluation frequency and epoch settings to the training loop.
- Evaluate the held-out test split only after model selection.

### 2. Define and write run artifacts

- Save the resolved configuration.
- Record source-data identity and split boundaries.
- Store metrics in a machine-readable tabular format.
- Save loss curves without requiring an interactive display.
- Make artifact writes explicit and testable.

### 3. Add checkpointing and resume behavior

- Save best-model checkpoints based on validation loss.
- Save periodic model, optimizer, epoch, step, token-count, and random-state
  checkpoints.
- Restore the complete training state.
- Verify resumed training against an uninterrupted tiny run.

### 4. Expose the installed command

- Add the project.scripts entry point only when the command performs the real
  workflow.
- Translate expected user errors into concise CLI messages and nonzero exit
  codes.
- Add an integration test that invokes the installed command.
- Run a tiny end-to-end CPU training job in a temporary directory.

### 5. Refine packaging boundaries

- Separate optional fine-tuning and TensorFlow weight-import dependencies from
  the core pre-training installation.
- Confirm source and wheel artifacts contain every required runtime file.
- Repeat isolated wheel installation and CLI smoke tests.

## Completed foundations

- Modern src-layout package built with pyproject.toml, uv, and uv_build.
- Verified source distributions, wheels, metadata, and isolated installation.
- Hermetic, warning-free unit-test baseline using tiny model configurations.
- Runtime-validated ModelConfig, TrainingConfig, and DataConfig dataclasses.
- Split token sequences before constructing overlapping language-model
  windows.
- Lazy token datasets and reproducible train, validation, and test loaders.
- Clean Ruff linting and formatting plus ty static type checking.
- Version-controlled VS Code configuration using the project environment.

## Working rules

- Keep operational paths outside TOML configuration.
- Validate inputs before creating persistent artifacts.
- Refuse implicit overwrite of an existing run directory.
- Keep unit tests hermetic and use integration tests for assembled workflows.
- Use rule-specific type suppressions only for deliberate invalid-input tests.
- Update this roadmap when work changes state; update an engineering note when
  a reusable lesson or decision changes.

## Related references

- [Observable workflow](engineering-notes/01-observable-workflows.md)
- [Runtime configuration](engineering-notes/12-runtime-configuration.md)
- [Verification commands](engineering-notes/14-verification-commands.md)
- [Typed TOML configuration](engineering-notes/18-typed-toml-configuration.md)
- [Typed negative tests and mocking](engineering-notes/19-typed-negative-tests-and-mocking.md)

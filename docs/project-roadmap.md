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

## Completed stage: assemble the pre-training application boundary

Completed in this stage:

- Added a strict TOML loader that constructs a frozen RunConfig.
- Added a checked-in tiny CPU configuration example.
- Added typed parsing for configuration, input, and output paths.
- Added run preparation that validates input files, loads configuration, and
  creates a fresh output directory without leaving partial artifacts on
  failure.
- Added hermetic unit tests for the parser and preparation boundary.
- Added explicit and automatic CPU, CUDA, and MPS device resolution with
  availability checks and deterministic fallback priority.
- Added bounded, deterministic seeding for Python, NumPy, and PyTorch random
  sources.
- Attached the effective device and seed as frozen ResolvedRuntime state to
  PreparedPretrainRun.
- Verified that missing inputs, invalid configuration, existing output paths,
  and device-resolution failures stop before seeding or creating output.

## Completed in the executable training workflow

- Added UTF-8 text loading with empty-input and file-error validation.
- Preserved text whitespace when encoding and made special-token handling
  explicit.
- Required matching model and GPT-2 tokenizer vocabularies.
- Added hermetic tokenizer mocks and regression cases for whitespace and
  literal special-token spellings.
- Connected text loading to the existing split and data-loader factory through
  create_pretrain_loaders.
- Verified configuration forwarding, returned bundle identity, and failure
  propagation with mocked dependencies.
- Added PretrainComponents and constructed the configured model on its
  resolved device before creating AdamW with explicit zero weight decay.
- Verified optimizer parameter identity, settings, and device movement using
  a tiny real CPU model, a method spy, and a shared prepared-run fixture.
- Computed token-weighted loader loss, including retained partial batches.
- Added strict evaluation-limit validation and explicit failure when no
  target tokens are evaluated, with small synthetic regression tests.
- Typed evaluate_model and restored its original mode after success or failure,
  while computing both losses in evaluation mode with gradients disabled.
- Verified distinct loader forwarding, loss ordering, and restoration for
  both starting modes and both possible loss-failure positions.
- Added a typed training loop with completed-update and target-token counters,
  scheduled evaluation, and a final evaluation without duplicate records.
- Verified real parameter updates, partial-batch token counts, evaluation
  records, and rejection of empty training before optimization.
- Added train_eval over the same training dataset, using sequential sampling
  and its own seeded generator.
- Gave each evaluation loader a separate explicit generator and verified that
  evaluation preserves the training generator and global PyTorch RNG state.
- Extended configured-settings, partial-batch, shape, and dtype checks to all
  four loaders.
- Assembled preparation, loaders, model/optimizer construction, and training
  through run_pretraining, returning the training result in memory.
- Verified stage order, exact argument forwarding, returned result identity,
  and failure propagation at each stage using autospecced mocks.
- Ran the real in-memory workflow through a tiny CPU integration test, including
  TOML loading, GPT-2 tokenization, optimization, and finite evaluation metrics.
- Configured pytest importlib mode so matching unit and integration test
  filenames are collected together without module-name collisions.

## Current task: persist run configuration and runtime metadata

Define the first run artifact: a JSON record of validated configuration and
effective runtime settings. Preserve requested values such as device="auto"
alongside the resolved device string. Make serialization and writing testable
before connecting them to workflow execution. Dataset identity, metric files,
model selection, and checkpoint persistence follow afterward.

## Next milestones

### 1. Build the executable training workflow

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
- Local pre-commit hooks and GitHub Actions checks for both supported Python
  versions; merge protection must be configured separately on GitHub.

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
- [Pre-commit and CI](engineering-notes/20-pre-commit-and-continuous-integration.md)
- [Resolved runtime and preparation order](engineering-notes/21-resolved-runtime-and-preparation.md)
- [Text loading and tokenizer contracts](engineering-notes/22-text-loading-and-tokenizer-contracts.md)
- [Workflow composition and mock results](engineering-notes/23-workflow-composition-and-mock-results.md)
- [Model, optimizer, and device tests](engineering-notes/24-model-optimizer-and-device-tests.md)
- [Token-weighted loss and empty evaluation](engineering-notes/25-token-weighted-loss-and-empty-evaluation.md)
- [Mode-preserving evaluation](engineering-notes/26-mode-preserving-evaluation.md)
- [Typed training progress and scheduling](engineering-notes/27-typed-training-progress-and-scheduling.md)
- [Evaluation-loader RNG isolation](engineering-notes/28-evaluation-loader-rng-isolation.md)
- [Assembled workflow and cross-function mocks](engineering-notes/29-assembled-workflow-and-cross-function-mocks.md)
- [Real workflow integration and test imports](engineering-notes/30-real-workflow-integration-and-test-imports.md)

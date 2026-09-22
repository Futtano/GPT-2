# Foundation checkpoint

This document records the state reached after the initial packaging and test
reliability work. It is a checkpoint, not a claim that the training product is
finished.

## Completed

- Documented the first training workflow and acceptance criteria.
- Established a modern `src/` package using `pyproject.toml` and `uv_build`.
- Built and inspected source and wheel distributions.
- Imported the wheel from an isolated environment outside the repository.
- Corrected package metadata and refreshed the lockfile.
- Removed network access from unit tests.
- Replaced an expected failure with explicit input validation.
- Removed the tensor-copy warning and tested storage ownership.
- Replaced full GPT-2 models in unit tests with small configurations.
- Added a frozen, runtime-validated `ModelConfig`.
- Wired typed configuration through model internals while preserving dictionary
  compatibility at public constructors.
- Added a frozen, runtime-validated `TrainingConfig` with explicit step-based
  frequencies, finite learning-rate checks, seed validation, and constrained
  device names.
- Added a frozen, runtime-validated `DataConfig` with explicit split fractions,
  a derived test fraction, stride, and worker count.
- Added deterministic token-sequence splitting before window construction,
  including actual-size validation and stable list outputs.
- Added a memory-efficient token dataset that constructs shifted windows lazily
  and retained the text-based dataset API as a compatibility wrapper.
- Extracted reusable validators for positive integers, non-negative integers,
  bounded finite real numbers, and Booleans.
- Organized unit tests to mirror the `gpt_2` package.
- Reached 235 passing tests with warnings treated as errors.

## Still planned

- Build separate training, validation, and test loaders from the token splits.
- Load configuration from a user-supplied file.
- Expose the pre-training workflow through an installed CLI.
- Save resolved configuration, metrics, and dataset metadata.
- Save best-model and resumable training checkpoints.
- Add integration and end-to-end tests.
- Separate optional feature dependencies from the core installation.

## Current acceptance target

A tiny CPU run should start from the installed CLI, use explicit input and
output paths, create reproducible splits, save inspectable artifacts, reload
the best model for generation, and resume training from saved state.

## Related notes

- [Observable workflows](01-observable-workflows.md)
- [Runtime configuration](12-runtime-configuration.md)
- [Verification commands](14-verification-commands.md)

[Back to the engineering-notes index](../engineering-notes.md)

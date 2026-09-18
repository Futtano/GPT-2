# Engineering notes

This is a living quick reference for the engineering lessons learned while
turning this repository from a collection of GPT-2 experiments into a
reproducible Python package. Update it when a section of the project is
completed or an earlier decision changes.

## 1. Define observable behavior before implementation

"Production-grade" is too broad to serve as an implementation target. Convert
it into behavior that a user or automated test can observe.

For the first pre-training workflow, success means that a user can:

1. Supply a local text file, configuration, and output directory.
2. Complete a tiny training run on CPU.
3. Find the resolved configuration, split metadata, metrics, and checkpoints
   in the selected output directory.
4. Load the best checkpoint for generation.
5. Resume from a training checkpoint with optimizer state and progress intact.
6. Evaluate on test data that was not used for training or model selection.

This contract keeps implementation choices tied to user-visible outcomes. It
also provides the basis for end-to-end acceptance tests later.

## 2. Do not couple installed packages to a source checkout

A relative path is resolved from the process's current working directory, not
from the repository that originally contained the package:

```python
dataset_path = Path("inputs/training.txt")
```

If a user runs an installed command from `~/experiments`, that path means
`~/experiments/inputs/training.txt`. The original repository may not exist on
the user's machine at all.

Accept important paths explicitly:

```bash
gpt2-train \
  --data ~/datasets/tiny-shakespeare.txt \
  --output-dir ~/experiments/run-001
```

Repository-level `inputs/` and `outputs/` directories may be documented as
conventions, but package behavior should not depend on them.

## 3. Split sequential data before creating windows

Overlapping language-model windows can leak nearly identical samples across
dataset splits. Given 1,000 tokens, split the sequence first:

```text
Training:   tokens   0-799
Validation: tokens 800-899
Test:       tokens 900-999
```

Then create windows independently within each section:

```python
train_windows = make_windows(tokens[:800])
val_windows = make_windows(tokens[800:900])
test_windows = make_windows(tokens[900:])
```

Creating windows first and randomly splitting afterward could put these two
samples in different splits:

```text
Training sample:   tokens 0-99
Validation sample: tokens 1-100
```

They share 99 tokens, so validation results would overstate generalization.

Use the splits for distinct purposes:

- **Training:** update model parameters.
- **Validation:** select checkpoints and make development decisions.
- **Test:** perform final evaluation after model selection.

## 4. Understand the Python packaging artifacts

Build the project with:

```bash
uv build
```

This project produces two distribution archives:

- **Source distribution (`.tar.gz`):** contains source code and the metadata
  needed to build the project.
- **Wheel (`.whl`):** contains the project in an installable layout. Installing
  it does not require building this project from source.

A wheel does not bundle its dependencies. It records dependency requirements
in metadata, and the installer resolves those packages separately. A wheel can
contain compiled extensions belonging to the project, but this project's
`py3-none-any` tag identifies a pure-Python, platform-independent wheel.

File placement alone does not decide archive contents. The build backend's
include and exclude rules do. Files outside `src/`, such as `README.md` and
`pyproject.toml`, can appear in a source distribution.

Useful inspection commands:

```bash
tar -tzf dist/gpt_2-0.1.0.tar.gz
unzip -l dist/gpt_2-0.1.0-py3-none-any.whl
unzip -p dist/*.whl '*/METADATA' | sed -n '1,30p'
```

Verify installation outside the repository so the source tree cannot hide a
packaging mistake:

```bash
uv venv /tmp/gpt2-wheel-check --python 3.13
uv pip install \
  --python /tmp/gpt2-wheel-check/bin/python \
  --no-deps \
  dist/gpt_2-0.1.0-py3-none-any.whl

cd /tmp
/tmp/gpt2-wheel-check/bin/python -c \
  "import gpt_2; from importlib.metadata import version; print(gpt_2.__file__); print(version('gpt-2'))"
```

`--no-deps` makes this a packaging smoke test. It does not prove that runtime
dependencies are complete or that model code works.

## 5. Treat `pyproject.toml` as part of the product

Package metadata affects installation and discoverability. At minimum, review:

- A concrete one-sentence description.
- The supported Python range.
- Direct runtime dependencies.
- Development dependencies.
- Optional dependencies for features that are not part of the core workflow.
- The build backend and its supported version range.

### Direct, transitive, development, and optional dependencies

- A **direct runtime dependency** is imported or otherwise required by
  installed package behavior.
- A **transitive dependency** is installed because one of the direct
  dependencies needs it. Do not declare it unless this project uses it
  directly.
- A **development dependency** is needed for tests, formatting, linting, or
  other contributor workflows.
- An **optional dependency** supports a feature that core users should not
  have to install.

The source audit found these direct imports:

```text
Core/model code:       torch, tiktoken
Weights/downloading:   numpy, tensorflow, requests, tqdm
Classification:        pandas, matplotlib
Plotting:              matplotlib
Instruction scoring:   tqdm
```

`requests` was missing from the metadata even though package code imported it.
`watermark` was not imported by package code or tests and was removed. `tqdm`
could not be removed because installed modules use it.

The current dependency list is intentionally honest about the current module
layout. Heavy feature-specific dependencies such as TensorFlow should become
optional only after module boundaries prevent core imports from requiring
them accidentally.

Audit imports with:

```bash
rg '^(from|import) ' src/gpt_2
```

### Choosing supported Python versions

Base the policy on:

1. The intersection of versions supported by direct dependencies.
2. The versions exercised continuously in CI.
3. Language and standard-library features the code uses.
4. The environments used by the intended audience.
5. The maintenance cost of every additional test environment.

A lower minimum supports more users but increases compatibility work. A newer
minimum simplifies code and dependency resolution but excludes older
environments.

Avoid an upper bound merely because a future Python version has not been
tested. An upper bound forces a new release before users can even try that
version. Use one when a known dependency or incompatibility makes it accurate.
The current `>=3.12,<3.14` range reflects the project's selected test range and
its current TensorFlow constraint. Reconsider it after TensorFlow becomes
optional and Python 3.14 is tested.

The Scientific Python ecosystem's SPEC 0 provides a useful time-based policy:
support Python feature releases for roughly three years after release.

### Build-backend constraints

The build requirement should include the backend used successfully by the
project:

```toml
[build-system]
requires = ["uv_build>=0.12.12,<0.13"]
build-backend = "uv_build"
```

The lower bound records a known working backend. uv treats minor releases as
the boundary at which breaking changes may occur, so `<0.13` prevents an
unreviewed backend upgrade. Patch releases within 0.12 remain allowed.

After metadata changes, refresh and verify both resolved dependencies and
built metadata:

```bash
uv lock
uv lock --check
uv build
unzip -p dist/*.whl '*/METADATA' | sed -n '1,30p'
```

## 6. Make unit tests hermetic

A hermetic unit test does not depend on network availability, external service
state, or files outside its temporary workspace.

Originally, dataset fixtures downloaded a text from GitHub. The same tests
passed with internet access and produced five failures or errors without it.
This is a classic flaky-test boundary: tests for token windows were also
implicitly testing HTTP and GitHub availability.

Separate responsibilities:

```text
Dataset tests       -> token windows and batch behavior
Downloader tests    -> download and cache behavior
```

Use short local text for dataset tests. Test the downloader by replacing
`urllib.request.urlretrieve` with a fake using pytest's `monkeypatch` fixture:

```python
def test_downloads_missing_file(tmp_path, monkeypatch):
    destination = tmp_path / "dataset.txt"
    calls = []

    def fake_retrieve(url, file_path):
        calls.append((url, file_path))
        file_path.write_text("fake dataset", encoding="utf-8")
        return str(file_path), None

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)

    result = download_pt_dataset(destination)

    assert len(calls) == 1
    assert result == destination
```

For cached behavior, install a fake that fails if called:

```python
def fail_if_called(*args, **kwargs):
    pytest.fail("urlretrieve should not be called for an existing file")
```

Mock at the boundary where the side effect occurs, and assert meaningful
interactions: call count, URL, destination, resulting contents, and cache
behavior.

## 7. Preserve APIs while improving tests

Test refactoring should not silently redesign production APIs. A downloader
test initially changed the function from:

```python
download_pt_dataset(file_path="...") -> path
```

to a required argument returning `(path, was_cached)`. That broke notebook
callers and changed the return type. The hermetic test did not require either
change, so the original contract was restored.

Before changing an API:

1. Search all callers with `rg`.
2. Decide whether the change is part of the current objective.
3. Update every caller and document the migration when it is intentional.
4. Add a behavioral test for the new contract.

For path-like inputs, preserve the caller's value unless normalization is an
explicit part of the API. A supplied `Path` should not become a string as an
accidental consequence of using another function's return value.

## 8. Assert expected failures directly

Use `xfail` for a known defect or unsupported behavior that is intentionally
still unresolved. Do not use it when the implementation already rejects an
invalid input as designed.

Instead of marking an invalid attention configuration as an expected failure,
make the contract explicit:

```python
with pytest.raises(
    ValueError,
    match="d_out must be divisible by num_heads",
):
    MultiHeadAttention(...)
```

For public input validation, prefer an explicit exception over `assert`:

```python
if d_out % num_heads != 0:
    raise ValueError("d_out must be divisible by num_heads")
```

Python may remove assertions when run with optimization, so assertions are
better suited to internal invariants than validation of user configuration.

## 9. Treat warnings as defects or explicit expectations

The suite exposed a PyTorch warning caused by copying an existing tensor with
`torch.tensor(source)`. The assignment helper must accept pretrained NumPy
arrays and tensor sources while matching the destination model parameter.

The required contract is:

- `left` supplies the required shape, dtype, and device.
- `right` supplies values and may be a tensor or NumPy array.
- The returned value is a trainable `torch.nn.Parameter`.
- The parameter owns its storage; later source mutation cannot change it.

The implementation pattern is:

```python
right_tensor = torch.as_tensor(
    right,
    dtype=left.dtype,
    device=left.device,
)
right_tensor = right_tensor.detach().clone()
return torch.nn.Parameter(right_tensor)
```

Each operation has a purpose:

- `as_tensor`: accepts both tensors and NumPy arrays and performs requested
  dtype/device conversion.
- `detach`: disconnects the source autograd history.
- `clone`: gives the parameter independent storage.
- `Parameter`: registers trainable model state with `requires_grad=True`.

Tests should challenge the contract rather than repeat matching defaults. Use
different source and destination dtypes, verify device and `requires_grad`,
then mutate both tensor and NumPy sources and confirm the parameter is
unchanged.

Run the suite with warnings promoted to failures:

```bash
uv run pytest -q -W error
```

## 10. Keep unit-test models tiny

Most unit tests verify shapes, finite values, loss behavior, and control flow.
They do not need the statistical capacity of GPT-2 124M.

The original tests repeatedly instantiated the full model. Replacing it with
small configurations produced these representative improvements:

```text
Training call:         6.63s -> about 0.4s
GPT model test:        2-3s  -> about 0.03s
Warm full test suite: 13.35s -> 5.80s
```

The pre-training test configuration retains GPT-2's full vocabulary because it
uses real `tiktoken` token IDs, while reducing model depth and width:

```python
TINY_GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 32,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
```

Pure model-shape tests generate token IDs directly, so they can also reduce the
vocabulary:

```python
TEST_GPT_CONFIG = {
    "vocab_size": 128,
    "context_length": 16,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}
```

Use function-scoped model fixtures when a test trains or otherwise mutates the
model. A shared model could make results depend on test execution order.

Measure optimization rather than assuming it worked:

```bash
uv run pytest -q --durations=10
```

## 11. Test assertions must be capable of failing

Watch for assertions that are always true or inspect the wrong dimension.

For a tensor shaped `(1, 0)`, `len(tensor)` is `1` because `len` reports the
first dimension. To test that it contains no token IDs, use:

```python
assert token_ids.shape == (1, 0)
assert token_ids.numel() == 0
```

Also remember that adjacent Python string literals concatenate automatically:

```python
[
    "Hello",
    "",       # the comma is required
]
```

Without the comma, the empty-string test case silently disappears.

Other useful assertion principles:

- Prefer `torch.equal` or `torch.allclose` to indirect Boolean expressions.
- Check finiteness when infinity would make a simple non-negative check pass.
- Avoid vacuous conditions such as `len(values) >= 0`.
- Verify state changes when testing training, not only that a function returns.
- Test error type and useful message for public validation.

## 12. Current verification commands

Use these after relevant changes:

```bash
# Dependency metadata
uv lock --check

# Distribution build
uv build

# Focused test while iterating
uv run pytest tests/test_module.py -q -W error

# Full suite with warnings treated as failures
uv run pytest -q -W error --durations=10

# Whitespace errors in the current diff
git diff --check
```

Run focused tests during iteration, then run the full suite once the focused
behavior passes. Do not repeatedly run expensive checks without a new change or
unresolved result that justifies them.

## 13. Completed foundation checkpoint

At this checkpoint, the project has:

- A documented first training workflow and acceptance criteria.
- A modern `src/` package built through `pyproject.toml` and `uv_build`.
- Verified source and wheel distributions.
- A wheel successfully imported from an isolated environment.
- Corrected package metadata and a consistent lockfile.
- Tests that do not require network access.
- Explicit invalid-configuration behavior instead of an expected failure.
- A warning-free tensor assignment path with storage-ownership coverage.
- Small test models that provide much faster feedback.
- 50 passing tests with warnings treated as errors at the latest checkpoint.

The next planned section introduces typed, validated model configuration before
wiring configuration files and a training CLI.

## Further reading

### Packaging

- [Python Packaging User Guide: Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [Python Packaging User Guide: The Packaging Flow](https://packaging.python.org/en/latest/flow/)
- [Python Packaging User Guide: Writing `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [uv: Building distributions](https://docs.astral.sh/uv/concepts/projects/build/)
- [uv: Build backend](https://docs.astral.sh/uv/concepts/build-backend/)

### Testing

- [pytest: How to monkeypatch and mock modules and environments](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)
- [pytest: How to write and report assertions](https://docs.pytest.org/en/stable/how-to/assert.html)
- [pytest: How to use skip and xfail](https://docs.pytest.org/en/stable/how-to/skipping.html)
- [pytest: How to capture warnings](https://docs.pytest.org/en/stable/how-to/capture-warnings.html)

### PyTorch tensor semantics

- [PyTorch: `torch.as_tensor`](https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html)
- [PyTorch: `torch.clone`](https://docs.pytorch.org/docs/stable/generated/torch.clone.html)
- [PyTorch: `Tensor.detach`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)
- [PyTorch: `torch.nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

### Compatibility policy

- [Scientific Python SPEC 0: Minimum Supported Dependencies](https://scientific-python.org/specs/spec-0000/)
- [Python Packaging User Guide: Supporting multiple Python versions](https://packaging.python.org/en/latest/guides/supporting-multiple-python-versions/)

# Treat `pyproject.toml` as part of the product

Package metadata affects installation, compatibility, and discoverability.
Review the description, Python range, dependencies, and build backend with the
same care as runtime code.

## Classify dependencies

- A **direct runtime dependency** is imported or otherwise required by
  installed behavior.
- A **transitive dependency** is installed because a direct dependency needs
  it. Do not declare it unless this project uses it directly.
- A **development dependency** supports tests, formatting, linting, or other
  contributor workflows.
- An **optional dependency** supports a feature that core users should not be
  forced to install.

The repository audit found these feature groups:

```text
Core/model code:       torch, tiktoken
Weights/downloading:   numpy, tensorflow, requests, tqdm
Classification:        pandas, matplotlib
Plotting:              matplotlib
Instruction scoring:   tqdm
```

`requests` was initially missing even though installed code imported it.
`watermark` was unused by package code and tests, so it was removed. Heavy
dependencies such as TensorFlow should become optional only after module
boundaries allow core imports to work without them.

Audit imports with:

```bash
rg '^(from|import) ' src/gpt_2
```

## Choose Python support deliberately

Use the intersection of dependency support, continuously tested versions,
language features, user environments, and available maintenance effort. A
lower minimum reaches more users but costs more compatibility work.

Avoid an upper bound merely because a future version is untested. Use one when
a known dependency or incompatibility makes it accurate. The current
`>=3.12,<3.14` range reflects the selected test range and TensorFlow constraint;
reconsider it after TensorFlow becomes optional and Python 3.14 is tested.

## Bound the build backend

```toml
[build-system]
requires = ["uv_build>=0.12.12,<0.13"]
build-backend = "uv_build"
```

The lower bound records a known working backend. The upper bound prevents an
unreviewed minor release, which uv may use for breaking changes.

After metadata changes:

```bash
uv lock
uv lock --check
uv build
unzip -p dist/*.whl '*/METADATA' | sed -n '1,30p'
```

## Further reading

- [Python Packaging User Guide: Writing `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [uv: Build backend](https://docs.astral.sh/uv/concepts/build-backend/)
- [Scientific Python SPEC 0](https://scientific-python.org/specs/spec-0000/)

[Back to the engineering-notes index](../engineering-notes.md)

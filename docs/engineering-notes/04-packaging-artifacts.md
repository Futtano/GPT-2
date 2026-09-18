# Understand source distributions and wheels

Build this project with:

```bash
uv build
```

The build produces two distribution archives:

- A **source distribution** (`.tar.gz`) contains source code and the metadata
  needed to build the project.
- A **wheel** (`.whl`) contains the project in an installable layout. Installing
  it does not require building this project from source.

A wheel does not bundle its dependencies. It records dependency requirements
in metadata, and the installer resolves those packages separately. A wheel may
contain compiled extensions belonging to the project. This project's
`py3-none-any` tag identifies a pure-Python, platform-independent wheel.

The build backend decides which files enter each archive. Being outside `src/`
does not automatically exclude a file: source distributions commonly contain
`README.md` and `pyproject.toml`.

## Inspect the artifacts

```bash
tar -tzf dist/gpt_2-0.1.0.tar.gz
unzip -l dist/gpt_2-0.1.0-py3-none-any.whl
unzip -p dist/*.whl '*/METADATA' | sed -n '1,30p'
```

Test installation outside the repository so local source files cannot hide a
packaging error:

```bash
uv venv /tmp/gpt2-wheel-check --python 3.13
uv pip install \
  --python /tmp/gpt2-wheel-check/bin/python \
  --no-deps \
  dist/gpt_2-0.1.0-py3-none-any.whl

cd /tmp
/tmp/gpt2-wheel-check/bin/python -c \
  "import gpt_2; print(gpt_2.__file__)"
```

`--no-deps` makes this a packaging smoke test only. It does not prove that
runtime dependencies are complete or that model code works.

## Further reading

- [Python Packaging User Guide: The Packaging Flow](https://packaging.python.org/en/latest/flow/)
- [Python Packaging User Guide: Package Formats](https://packaging.python.org/en/latest/discussions/package-formats/)
- [uv: Building distributions](https://docs.astral.sh/uv/concepts/projects/build/)

[Back to the engineering-notes index](../engineering-notes.md)

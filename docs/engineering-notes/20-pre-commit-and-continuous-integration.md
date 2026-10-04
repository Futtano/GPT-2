# Enforce checks before commits and on GitHub

Editor diagnostics, commit hooks, and CI serve different points in the same
workflow. The editor gives immediate feedback. Hooks catch mistakes before a
local commit. CI checks the committed repository in a fresh environment.

## Install the local hooks

The repository stores hook definitions in `.pre-commit-config.yaml` and includes
pre-commit in its development dependencies. Each developer installs the actual
Git hook after cloning:

```bash
uv sync --locked --dev
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
```

Local hooks invoke Ruff and ty through `uv run --locked`, using the tool versions
in `uv.lock` and the policy in `pyproject.toml`. `--locked` fails if dependency
metadata requires a lockfile update instead of silently updating it.

Ruff lint fixes run before formatting. If a hook changes files, inspect the
changes, stage them again, and retry the commit. ty uses `pass_filenames: false`
to check its configured project scope, including callers of changed functions,
and `always_run: true` to cover dependency and configuration changes too.

Hook installation lives in `.git/hooks`, so it is local to each checkout. The
shared YAML file does not automatically install hooks for other developers.

## Check the committed project in GitHub Actions

The workflow in `.github/workflows/ci.yml` runs on pushes, pull requests, and
manual requests. Each Ubuntu job selects Python 3.12 or 3.13, installs the locked
project with development dependencies, and runs:

```bash
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv run --locked ty check
uv run --locked pytest -q -W error
```

CI reports failures instead of applying fixes. The matrix exercises both
declared Python versions; Ruff and ty still use the configured minimum version
as their analysis target.

The workflow pins action revisions and the uv version. Its token has read-only
repository access, it cancels superseded runs on the same ref, and each job has
a timeout. uv caches downloads to reduce repeated dependency setup.

Tests run on CPU; accelerator availability tests use mocks. The current lockfile
still includes CUDA and TensorFlow dependencies, so a fresh installation may be
large even though tests do not require GPU hardware. Dependency separation is a
later packaging milestone.

`MPLBACKEND=Agg` selects a noninteractive plotting backend. A separate setup
step initializes the GPT-2 tokenizer cache because tiktoken downloads encoding
assets on first use. This makes setup failures distinct from test failures; it
does not enforce network isolation during pytest.

## Require successful checks before merging

Local hooks can be skipped. GitHub Actions automatically reports failures, but
blocking merges requires a repository ruleset or branch-protection rule.

After the first CI run, require both statuses for the protected branch:

- `Checks (Python 3.12)`
- `Checks (Python 3.13)`

This setting is maintained on GitHub, separately from the workflow YAML. The
workflow alone does not prohibit merging a failing pull request.

## Further reading

- [pre-commit configuration and usage](https://pre-commit.com/)
- [Ruff hook integration](https://docs.astral.sh/ruff/integrations/)
- [uv in GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)
- [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)

[Back to the engineering-notes index](../engineering-notes.md)

# Make the development environment reproducible

Editor feedback is part of the development toolchain. If VS Code, a terminal,
and continuous integration use different tool versions or rules, developers
can see different results for the same checkout. A production-quality project
pins the command-line tools and stores their policy in version control.

## Give each tool one responsibility

This project uses:

- Ruff for linting and import sorting;
- Ruff for deterministic code formatting;
- ty for static type checking and Python language services;
- pytest for executable behavior.

Ruff and ty are development dependencies, so uv records their versions in
uv.lock:

~~~toml
[dependency-groups]
dev = [
    "pytest>=9.1.1",
    "pytest-mock>=3.15.1",
    "ruff>=0.16.9",
    "ty>=0.0.84",
]
~~~

After cloning or changing dependencies, run uv sync. Commands such as
uv run ruff and uv run ty then use the locked project environment instead of
an unrelated global installation.

## Store tool policy in pyproject.toml

The minimum supported Python version is the correct analysis target. The
package supports Python 3.12 and 3.13, so Ruff and ty target 3.12 even when the
developer happens to run Python 3.13.

Ruff owns a deliberately selected set of correctness and maintainability
rules:

~~~toml
[tool.ruff]
target-version = "py312"
line-length = 88
src = ["src"]
extend-exclude = ["notebooks"]

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I", "B", "C4", "UP", "SIM", "TRY"]
ignore = ["TRY003", "TRY004"]
~~~

Selecting rules explicitly prevents a user-level or parent-directory Ruff
configuration from silently changing project results. TRY003 and TRY004 are
excluded because this project keeps useful error messages at the raise site
and deliberately exposes ValueError consistently for invalid configuration.

ty checks package code and tests while leaving exploratory notebooks outside
the production gate:

~~~toml
[tool.ty.environment]
python-version = "3.12"
root = ["./src"]

[tool.ty.src]
include = ["src", "tests"]
exclude = ["notebooks"]
~~~

uv activates the project environment for command-line checks. ty can also
discover a .venv directory automatically, so an environment path does not
need to be hard-coded in pyproject.toml.

## Keep overrides narrow and explained

Runtime-validation tests intentionally pass values that contradict constructor
annotations. Those calls are invalid statically because their purpose is to
prove that runtime validation rejects them. A file-specific ty override
prevents this expected mismatch from hiding useful diagnostics elsewhere:

~~~toml
[[tool.ty.overrides]]
include = ["tests/unit/gpt_2/test_config.py"]

[tool.ty.overrides.rules]
invalid-argument-type = "ignore"
invalid-assignment = "ignore"
~~~

Prefer a narrow override or a rule-specific suppression over disabling a rule
for the whole project. The rest of the test suite remains fully checked.

## Make VS Code use the project tools

Installing extensions enables editor integration, but workspace settings still
need to select the project environment and define save behavior.

The repository settings:

- select .venv/bin/python;
- use Ruff as the Python formatter;
- format files on save;
- apply explicit Ruff fixes and import sorting on save;
- prefer pyproject.toml over personal Ruff settings;
- find Ruff and ty through the selected environment;
- enable workspace-wide ty diagnostics.

The ty extension normally disables the Python extension's language server so
that two Python language servers do not publish duplicate diagnostics. In this
project, ty provides completion, hover, navigation, and type checking. If a
team instead keeps Pylance for language features, it should set
ty.disableLanguageServices to true and use ty only for diagnostics.

The extensions file recommends the official ty, Ruff, and Python extensions to
new contributors. Extension recommendations improve setup but do not replace
the locked command-line dependencies used by local checks and CI.

## Run the same gates locally and in CI

~~~bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
uv run pytest -q -W error
~~~

Ruff linting, Ruff formatting, and ty type checking currently pass across the
configured source and test scope. A clean baseline makes every new diagnostic
attributable to the current change instead of an inherited backlog.

## Further reading

- [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
- [Ruff editor settings](https://docs.astral.sh/ruff/editors/settings/)
- [ty configuration](https://docs.astral.sh/ty/configuration/)
- [ty editor integration](https://docs.astral.sh/ty/editors/)
- [ty module discovery](https://docs.astral.sh/ty/modules/)
- [uv dependency management](https://docs.astral.sh/uv/concepts/projects/dependencies-and-groups/)

[Back to the engineering-notes index](../engineering-notes.md)

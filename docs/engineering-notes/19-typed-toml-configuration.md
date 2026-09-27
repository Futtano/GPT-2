# Load typed run configuration from TOML

A configuration file is untrusted input even when it belongs to the project.
Loading it should cross a clear boundary from loosely typed TOML values to
validated application objects.

Keep that boundary in two functions:

~~~python
def load_run_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as file:
        raw = tomllib.load(file)

    return parse_run_config(raw)
~~~

load_run_config owns file access and TOML decoding. parse_run_config owns
schema checks and construction of the typed configuration. This separation
allows most behavior to be tested with ordinary mappings, while a few focused
tests cover files and malformed TOML.

## Compose focused configuration objects

A run combines configuration objects that already own their local validation:

~~~python
@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
~~~

RunConfig represents one resolved application configuration. It should not
repeat field validation performed by ModelConfig, DataConfig, and
TrainingConfig.

dataclasses.asdict recursively converts the complete object tree into
dictionaries. This is useful for logging resolved settings and storing
configuration beside checkpoints. Test the complete nested value rather than
only checking that the result is a dictionary.

## Validate the outer schema explicitly

A strict root schema catches misspelled or obsolete sections early:

~~~python
required_sections = frozenset({"model", "data", "training"})
actual_sections = set(raw)

missing = required_sections - actual_sections
unknown = actual_sections - required_sections
~~~

Reject both missing and unknown sections. Silently accepting an unknown table
such as [traning] would make a typo look like a valid experiment.

Each required value must also be a mapping before it is expanded with **.
This produces a domain-specific message such as:

~~~text
[training] must be a TOML table.
~~~

The project deliberately exposes ValueError for all structurally or
semantically invalid configuration content. Callers such as a CLI can
therefore handle one configuration-error category. FileNotFoundError and
tomllib.TOMLDecodeError remain distinct because they describe file access and
TOML syntax failures.

## Add context without losing the original error

Dataclass construction can fail because a field is missing, unknown, or
invalid. Translate these low-level failures at the section boundary:

~~~python
try:
    training = TrainingConfig(**training_values)
except (TypeError, ValueError) as error:
    raise ValueError(
        f"Invalid [training] configuration: {error}"
    ) from error
~~~

The section name tells the user where to look. The raise-from form retains the
original exception as its cause, preserving useful traceback context.

## Keep parsing free of mutation

A parser should treat the supplied mapping as input. It should not delete,
normalize, or insert values unless mutation is part of its documented
contract. A focused regression test can protect this property:

~~~python
raw = {
    "model": VALID_MODEL_CONFIG.copy(),
    "data": VALID_DATA_CONFIG.copy(),
    "training": VALID_TRAINING_CONFIG.copy(),
}
expected = deepcopy(raw)

parse_run_config(raw)

assert raw == expected
~~~

This matters when callers reuse the decoded mapping for logging, comparison,
or diagnostics.

## Test the boundary in layers

A useful test set covers:

- a valid in-memory mapping;
- a valid TOML file written under pytest's tmp_path;
- missing and unknown root sections;
- a root section that is not a mapping;
- missing, unknown, and invalid fields within every section;
- preservation of the input mapping;
- recursive conversion with dataclasses.asdict;
- malformed TOML, which raises tomllib.TOMLDecodeError;
- a missing path, which raises FileNotFoundError.

Use setattr with a variable attribute name when testing a frozen dataclass.
The test intentionally performs an operation rejected at runtime, while the
indirection prevents static analysis and lint rules from treating the test
itself as an ordinary invalid assignment.

## Split files when responsibilities split

File length alone is a weak reason to reorganize code. A long test module may
still be cohesive when most lines are parameter tables for one API.

Keep configuration tests together while they share the same sample mappings
and exercise one construction boundary. Split them when another independent
responsibility appears, such as CLI argument overrides, environment-variable
merging, configuration migrations, or serialization formats. At that point,
names such as test_config_models.py, test_config_loading.py, and
test_config_overrides.py describe real behavioral boundaries.

## Further reading

- [Python documentation: tomllib](https://docs.python.org/3/library/tomllib.html)
- [Python documentation: exception chaining](https://docs.python.org/3/tutorial/errors.html#exception-chaining)
- [Python documentation: dataclasses.asdict](https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict)
- [pytest documentation: temporary directories and files](https://docs.pytest.org/en/stable/how-to/tmp_path.html)

[Back to the engineering-notes index](../engineering-notes.md)

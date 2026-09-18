# Engineering notes

These notes record the engineering lessons learned while turning this
repository from GPT-2 experiments into a reproducible Python package. Each
topic is a self-contained document with examples and focused references.

## Workflow and data

1. [Define observable behavior before implementation](engineering-notes/01-observable-workflows.md)
2. [Keep installed packages independent of the source checkout](engineering-notes/02-portable-paths.md)
3. [Split sequential data before creating windows](engineering-notes/03-sequential-data-splits.md)

## Packaging and compatibility

4. [Understand source distributions and wheels](engineering-notes/04-packaging-artifacts.md)
5. [Treat `pyproject.toml` as part of the product](engineering-notes/05-project-metadata.md)

## Reliable tests

6. [Make unit tests hermetic](engineering-notes/06-hermetic-tests.md)
7. [Preserve APIs while improving tests](engineering-notes/07-api-preservation.md)
8. [Assert expected failures directly](engineering-notes/08-expected-failures.md)
9. [Handle PyTorch tensor copying explicitly](engineering-notes/09-pytorch-tensor-copying.md)
10. [Keep unit-test models tiny](engineering-notes/10-fast-ml-tests.md)
11. [Write assertions that can fail meaningfully](engineering-notes/11-effective-assertions.md)

## Configuration and project structure

12. [Validate typed configuration at runtime](engineering-notes/12-runtime-configuration.md)
13. [Organize tests by scope and package ownership](engineering-notes/13-test-organization.md)

## Working reference

14. [Verification commands](engineering-notes/14-verification-commands.md)
15. [Foundation checkpoint](engineering-notes/15-foundation-checkpoint.md)

Update the relevant topic when a decision changes. Add a new document when a
completed section introduces a distinct engineering concept.

# Define observable behavior before implementation

“Production-grade” is too broad to guide implementation. Translate it into
behavior that a user or automated test can observe.

For this project's first pre-training workflow, success means that a user can:

1. Supply a local text file, configuration, and output directory.
2. Complete a tiny training run on CPU.
3. Find the resolved configuration, split metadata, metrics, and checkpoints
   in the selected output directory.
4. Load the best checkpoint for text generation.
5. Resume from a training checkpoint with optimizer state and progress intact.
6. Evaluate on test data that was not used for training or model selection.

These statements are acceptance criteria. They describe outcomes without
prematurely deciding how the CLI, configuration parser, or checkpoint format
must work. Once implementation begins, each criterion can become an
integration or end-to-end test.

When defining a workflow, identify four things:

- **Inputs:** data, configuration, checkpoint, and destination paths.
- **Processing:** the ordered stages that transform those inputs.
- **Artifacts:** files and metrics that remain after execution.
- **Evidence:** checks proving that the workflow is reproducible and correct.

This prevents a project from accumulating unrelated features while its primary
user journey remains incomplete.

## Further reading

- [The Twelve-Factor App: Processes](https://12factor.net/processes)
- [pytest: Good integration practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)

[Back to the engineering-notes index](../engineering-notes.md)

# Keep installed packages independent of the source checkout

A relative path is resolved from the process's current working directory, not
from the repository that originally contained the package:

```python
dataset_path = Path("inputs/training.txt")
```

If an installed command runs from `~/experiments`, this refers to
`~/experiments/inputs/training.txt`. The original repository may not exist on
the user's machine, and the installed package may live under `site-packages`.

Accept user data and output paths explicitly:

```bash
gpt2-train \
  --data ~/datasets/tiny-shakespeare.txt \
  --output-dir ~/experiments/run-001
```

Repository-level `inputs/` and `outputs/` directories can remain documented
conventions for local development. They should not be required by installed
package behavior.

Package-owned static resources are a separate case. Place those inside the
import package and read them with `importlib.resources`; do not locate them by
walking upward from `__file__` toward an assumed repository root.

## Quick checklist

- Accept external data and output paths from the user.
- Resolve and record paths at the application boundary.
- Create missing output directories deliberately.
- Avoid `Path.cwd()` as an implicit project-root locator.
- Use `importlib.resources` only for files shipped inside the package.

## Further reading

- [Python documentation: `pathlib`](https://docs.python.org/3/library/pathlib.html)
- [Python documentation: `importlib.resources`](https://docs.python.org/3/library/importlib.resources.html)

[Back to the engineering-notes index](../engineering-notes.md)

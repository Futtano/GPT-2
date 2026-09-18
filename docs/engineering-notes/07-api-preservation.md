# Preserve APIs while improving tests

Test refactoring should not silently redesign production APIs.

While making the downloader tests hermetic, the function was temporarily
changed from:

```python
download_pt_dataset(file_path="...") -> path
```

to a required argument returning `(path, was_cached)`. Existing notebooks
called it without an argument and expected a path. The test improvement did not
require an API change, so the original contract was restored.

Before changing an API:

1. Search all callers with `rg`.
2. Decide whether the change belongs to the current objective.
3. Update every caller and document the migration if it is intentional.
4. Add a behavioral test for the new contract.

Preserve path-like values unless normalization is explicit. A supplied `Path`
should not accidentally become a string because another function returns one:

```python
urllib.request.urlretrieve(url, file_path)
return file_path
```

Avoid assigning the downloader's return value back to `file_path` when the API
promises to return the caller's original destination object.

## Further reading

- [Semantic Versioning](https://semver.org/)
- [Python Packaging User Guide: Versioning discussion](https://packaging.python.org/en/latest/discussions/versioning/)

[Back to the engineering-notes index](../engineering-notes.md)

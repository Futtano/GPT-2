# Make unit tests hermetic

A hermetic unit test does not depend on network availability, external service
state, or files outside its temporary workspace.

The original dataset fixture downloaded text from GitHub. It passed with
internet access and produced five failures or errors without it. Dataset tests
were therefore testing GitHub and HTTP in addition to token windows.

Separate responsibilities:

```text
Dataset tests       -> token windows and batch behavior
Downloader tests    -> download and cache behavior
```

Use local text for dataset tests. Replace network access in downloader tests
with a fake through pytest's `monkeypatch` fixture:

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

For cached behavior, make any attempted download fail the test:

```python
def fail_if_called(*args, **kwargs):
    pytest.fail("urlretrieve should not be called for an existing file")
```

Assert the interaction that matters: call count, URL, destination, resulting
contents, and cache behavior. A fake that only prevents network traffic without
checking its inputs can allow incorrect downloader behavior to pass.

## Further reading

- [pytest: Monkeypatching functions and environments](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)
- [Python documentation: `unittest.mock`](https://docs.python.org/3/library/unittest.mock.html)

[Back to the engineering-notes index](../engineering-notes.md)

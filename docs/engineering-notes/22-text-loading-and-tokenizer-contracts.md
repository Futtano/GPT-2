# Load text without silently changing the training data

`load_text_token_ids` in `dataset.py` turns a local UTF-8 text file into a list
of GPT-2 token IDs. File validation, decoding, tokenizer selection, and
vocabulary compatibility belong at this boundary, before dataset construction.

## Validate content without trimming it

Check that the path exists and identifies a file, then decode with UTF-8 and
strict error handling. Missing files, directories, and invalid UTF-8 fail
before tokenization.

Use stripping only to decide whether text has meaningful content:

```python
txt = path.read_text(encoding="utf-8", errors="strict")
if not txt.strip():
    raise ValueError("Input is empty or contains only whitespace.")
```

Pass `txt` itself to the tokenizer. Assigning `txt = txt.strip()` would remove
leading and trailing spaces, tabs, or newlines and alter the training tokens.
This boundary performs no explicit whitespace cleanup. Standard text-file
reading can still normalize line endings; it is not a byte-preserving reader.

## Keep the model and tokenizer vocabulary compatible

For this GPT-2 workflow, require:

```python
model_config.vocab_size == tokenizer.n_vocab
```

The tokenizer produces IDs used as embedding indices and next-token targets.
A smaller model vocabulary cannot represent every tokenizer ID. A larger
model vocabulary gives the output layer classes that the tokenizer cannot
decode. Equality keeps the input, loss targets, and generated IDs compatible.

This is a workflow constraint. Isolated model tests can continue using small
synthetic vocabularies when they do not use the GPT-2 tokenizer.

## Make special-token handling explicit

The current workflow treats literal special-token spellings as ordinary text:

```python
tokenizer.encode(txt, disallowed_special=())
```

For example, the string `"Hello <|endoftext|>"` is encoded as ordinary text
rather than rejected or converted into the special end-of-text token. This
setting does not append end-of-text tokens or add document boundaries.

## Verify arguments as well as results

A fake encoder that always returns `[1, 2, 3, 4]` can hide incorrect input
handling. A mock lets the success test check the returned IDs and the exact
text and policy sent to the encoder:

```python
tokenizer = mocker.Mock()
tokenizer.n_vocab = model_config.vocab_size
tokenizer.encode.return_value = expected
get_encoding = mocker.patch(
    "gpt_2.dataset.tiktoken.get_encoding",
    return_value=tokenizer,
)

result = load_text_token_ids(path, model_config=model_config)

assert result == expected
get_encoding.assert_called_once_with("gpt2")
tokenizer.encode.assert_called_once_with(txt, disallowed_special=())
```

Include ordinary text, leading and trailing whitespace, and a literal
special-token spelling in the valid-input cases. The whitespace case catches
accidental trimming. On vocabulary mismatch, assert that `encode` was never
called. Invalid-file and empty-input cases exercise failures before tokenizer
construction.

These unit tests use temporary local files and mock tokenizer construction, so
the new boundary tests need neither network access nor tokenizer downloads.
The assertions verify orchestration; real tokenizer behavior belongs in an
integration test.

## Avoid accidental fake and fixture behavior

An instance method needs `self`. A fake `encode(text, *args, **kwargs)` method
without it receives the instance as `text`; permissive `*args` can hide the
mistake by accepting the actual string as another argument.

pytest supplies a fixture value when its name appears as a test parameter.
Referencing the module-level fixture name without requesting it refers to the
fixture definition, not its returned value. A vocabulary-rejection test does
not need an encoder return value because encoding should never be reached.

## Further reading

- [Python Path.read_text](https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text)
- [tiktoken implementation and special-token policy](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py)
- [pytest fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [Mock call assertions](https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.assert_called_once_with)

[Back to the engineering-notes index](../engineering-notes.md)

# Describe dataset identity and exact token split boundaries

`build_dataset_metadata` receives source bytes, their token IDs, and DataConfig
and returns a JSON-compatible record. It does not read a file, tokenize text,
write artifacts, or construct windows.

## Identify contents and tokenization policy

The version-one payload records SHA-256 over raw source bytes, byte length,
tokenizer name `gpt2`, special-token policy `ordinary_text`, total token count,
and start, exclusive end, and count for each split.

Hash contents rather than a filename: changed bytes have a different identity
even if their path stays the same. Byte length, character count, and token
count are separate quantities.

The builder assumes its bytes and token IDs correspond. Later integration must
hash and tokenize the same file read to enforce that relationship.

## Reuse split rounding and validation

The builder calls `split_token_ids` and derives contiguous boundaries from the
returned lengths. Training and validation lengths are rounded down separately:

```text
For 13 tokens and fractions 0.6 and 0.2:
training length   = int(13 * 0.6) = 7
validation length = int(13 * 0.2) = 2
validation end    = 7 + 2 = 9
test length       = 13 - 9 = 4
```

`int(13 * (0.6 + 0.2))` would give 10, a different boundary. The regression
test now calculates expected lengths separately and includes 13 tokens to
expose that mistake.

End positions are exclusive, like Python slices. Seven tokens yield:

| Split | Start | Exclusive end | Count |
| --- | --- | --- | --- |
| Train | 0 | 4 | 4 |
| Validation | 4 | 5 | 1 |
| Test | 5 | 7 | 2 |

These positions are token offsets, not token values, byte offsets, or window
indices. Windows are built after splitting.

## Make unit inputs expose the contract

Metadata tests can use arbitrary integer token IDs paired with unrelated
bytes. This pure builder performs no tokenization. Non-UTF-8 bytes are valid
hash inputs; decoding checks belong to the text-loading boundary.

Whole-payload tests use range, list, and tuple inputs. Changed-source tests
verify that the digest itself changes, not merely the surrounding dictionary
whose byte-count field might differ. Boundary tests check adjacency, positive
lengths, count arithmetic, full coverage, and token-sequence reconstruction.
An unsorted mutable token list verifies the builder does not modify its input.

## Reach the intended validation layer

Invalid fractions fail during DataConfig construction, before the builder is
called. To test empty-split propagation, supply valid fraction configurations
with too few tokens. Three tokens with fractions 0.1 and 0.4 produce an empty
training split; fractions 0.6 and 0.1 produce an empty validation split.

The tests now assert the builder propagates the existing `Empty splits`
ValueError, rather than accidentally failing while creating configuration.

## Completion and next boundary

The full suite passed with 380 tests, with Ruff linting, formatting, and ty
passing. The builder is complete; dataset metadata is not yet persisted.
Next, add a JSON writer with the existing exclusive-write policy, then connect
it to bytes obtained from the same read used for tokenization.

## Further reading

- [Python hashlib](https://docs.python.org/3/library/hashlib.html)
- [Sequential split policy](03-sequential-data-splits.md)
- [Text and tokenizer contracts](22-text-loading-and-tokenizer-contracts.md)
- [Exclusive JSON writes](31-run-metadata-and-exclusive-json-writes.md)

[Back to the engineering-notes index](../engineering-notes.md)

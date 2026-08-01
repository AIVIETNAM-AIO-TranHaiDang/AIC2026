"""Sentence chunking of long queries for the dense visual channel (note 15).

SigLIP-family text towers are trained on short single-sentence alt-text:
their position table is 64 tokens and, structurally, they attend mostly to
the first sentence of a multi-sentence input (arXiv 2602.22419, "CLIP Is
Shortsighted"). A 3-sentence KIS query fed whole therefore encodes mostly
sentence one — and a KIS query's sentences usually describe *consecutive
moments*, not one frame. Splitting the query per sentence and dispatching
each chunk as its own dense-visual sub-query lets every sentence find its
own moment; the existing temporal window fusion then rewards the window
that covers them all (and, with ``temporal.order_bonus``, in the right
order — vitrivr's temporal scoring, VBS).

The splitter is deliberately dumb: punctuation and newlines, no language
model. Vietnamese sentence boundaries follow the same terminal marks as
English, and the synthetic fixture generator is prompted for "two to four
sentences", so this covers the real query shapes.
"""

from __future__ import annotations

import re

# Terminal punctuation that ends a sentence. A library/format contract of
# written Vietnamese and English, not a tunable: the ellipsis and newline
# are included because moderators' KIS texts use both as separators.
_SENTENCE_BOUNDARY = re.compile(r"[.!?;…]+|\n+")


def split_query_sentences(text: str, max_chunks: int) -> list[str]:
    """Split a query into sentence chunks, at most ``max_chunks``.

    Returns ``[text]`` unchanged (stripped) when there is nothing to split
    — zero or one sentence — so callers can use ``len(result) > 1`` as the
    "was actually chunked" test. Sentences beyond ``max_chunks`` merge into
    the last chunk rather than being dropped: content is never lost, only
    granularity.
    """
    stripped = text.strip()
    if not stripped:
        return []
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(stripped)]
    sentences = [part for part in parts if part]
    if len(sentences) <= 1:
        return [stripped]
    if len(sentences) > max_chunks:
        head = sentences[: max_chunks - 1]
        tail = ". ".join(sentences[max_chunks - 1 :])
        sentences = head + [tail]
    return sentences

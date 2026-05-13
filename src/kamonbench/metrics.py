from __future__ import annotations

from collections.abc import Sequence

from .text import normalize_japanese_for_comparison


def edit_counts(reference: Sequence[str], predicted: Sequence[str]) -> tuple[int, int, int]:
    dp = [[0] * (len(predicted) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        dp[i][0] = i
    for j in range(len(predicted) + 1):
        dp[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(predicted) + 1):
            if reference[i - 1] == predicted[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    i, j = len(reference), len(predicted)
    insertions = deletions = substitutions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == predicted[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1

    return insertions, deletions, substitutions


def character_edit_distance(reference: str, predicted: str) -> int:
    ref_chars = list(normalize_japanese_for_comparison(reference))
    pred_chars = list(normalize_japanese_for_comparison(predicted))
    return sum(edit_counts(ref_chars, pred_chars))


def character_error_rate(reference: str, predicted: str):
    ref_chars = list(normalize_japanese_for_comparison(reference))
    pred_chars = list(normalize_japanese_for_comparison(predicted))

    insertions, deletions, substitutions = edit_counts(ref_chars, pred_chars)
    total_operations = insertions + deletions + substitutions
    error_rate = total_operations / max(len(ref_chars), 1) if len(ref_chars) > 0 else float('inf')
    return insertions, deletions, substitutions, error_rate


def normalized_tokens(text: str) -> list[str]:
    return [normalize_japanese_for_comparison(t) for t in text.split()]


def token_edit_distance(reference: str, predicted: str) -> int:
    return sum(edit_counts(normalized_tokens(reference), normalized_tokens(predicted)))


def token_error_rate(reference: str, predicted: str) -> tuple[int, int, int, float]:
    ref_tokens = normalized_tokens(reference)
    pred_tokens = normalized_tokens(predicted)
    insertions, deletions, substitutions = edit_counts(ref_tokens, pred_tokens)
    total_operations = insertions + deletions + substitutions
    error_rate = total_operations / max(len(ref_tokens), 1) if len(ref_tokens) > 0 else float('inf')
    return insertions, deletions, substitutions, error_rate

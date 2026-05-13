from __future__ import annotations

import jaconv


def normalize_japanese_for_comparison(text: str) -> str:
    text = text.replace(' ', '').replace('\t', '').replace('\n', '')
    return jaconv.kata2hira(text)


def normalize_english_whitespace(text: str) -> str:
    return ' '.join(text.split())


def strip_after_token(tokens: list[int], end_token: int) -> list[int]:
    try:
        end_pos = tokens.index(end_token)
        return tokens[:end_pos]
    except ValueError:
        return tokens


def tokens_to_text(tokens: list[int], label_to_expr: dict[int, str]) -> str:
    return ' '.join([label_to_expr.get(t, f'<UNK:{t}>') for t in tokens])

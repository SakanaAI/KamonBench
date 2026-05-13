from __future__ import annotations

from typing import Any

import jaconv

END_TOKEN = '<EOS>'


def analysis_tokens(record: dict[str, Any]) -> list[str]:
    exprs = []
    for item in record.get('analysis', []):
        expr = item.get('expr', '')
        if expr:
            exprs.append(jaconv.kata2hira(expr))
    return exprs


def translation_tokens(record: dict[str, Any]) -> list[str]:
    translation = record.get('translation', '')
    return translation.split() if translation else []


def create_label_sets(records: list[dict[str, Any]]):
    analysis_expressions: set[str] = set()
    for record in records:
        analysis_expressions.update(analysis_tokens(record))

    translation_expressions: set[str] = set()
    for record in records:
        translation_expressions.update(translation_tokens(record))

    analysis_vocab = sorted(analysis_expressions)
    analysis_vocab.append(END_TOKEN)
    translation_vocab = sorted(translation_expressions)
    translation_vocab.append(END_TOKEN)

    expr_to_label = {e: i for i, e in enumerate(analysis_vocab)}
    label_to_expr = {i: e for e, i in expr_to_label.items()}
    translation_expr_to_label = {e: i for i, e in enumerate(translation_vocab)}
    translation_label_to_expr = {i: e for e, i in translation_expr_to_label.items()}
    return expr_to_label, label_to_expr, translation_expr_to_label, translation_label_to_expr

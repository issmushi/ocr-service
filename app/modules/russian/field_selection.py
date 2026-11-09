"""Map public document fields to the OCR labels needed to produce them."""

from collections.abc import Mapping, Sequence


def ocr_fields_for(
    mapping: Mapping[str, str | tuple[str, ...] | None],
    fields: Sequence[str] | None,
) -> tuple[str, ...]:
    labels = []
    for field in mapping if fields is None else fields:
        source = mapping[field]
        if source is not None:
            labels.extend((source,) if isinstance(source, str) else source)
    return tuple(dict.fromkeys(labels))

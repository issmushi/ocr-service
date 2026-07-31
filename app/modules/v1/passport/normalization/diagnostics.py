from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

from app.modules.v1.passport.normalization.models import (
    NormalizedPassportHeader,
)

MRZ_FIELDS = {
    1: (
        (1, 2, "Тип документа"),
        (3, 5, "Страна выдачи"),
        (6, 44, "Фамилия и имена"),
    ),
    2: (
        (1, 9, "Номер паспорта"),
        (10, 10, "Контрольная цифра номера паспорта"),
        (11, 13, "Гражданство"),
        (14, 19, "Дата рождения"),
        (20, 20, "Контрольная цифра даты рождения"),
        (21, 21, "Пол"),
        (22, 27, "Дата окончания срока действия"),
        (28, 28, "Контрольная цифра срока действия"),
        (29, 42, "Дополнительные данные"),
        (43, 43, "Контрольная цифра дополнительных данных"),
        (44, 44, "Общая контрольная цифра"),
    ),
}


class MrzCorrection(TypedDict):
    line: int
    position: int
    source_position: int | None
    field: str
    operation: Literal["замена", "вставка"]
    source: str
    replacement: str
    reason: str


class MrzNormalizationIssue(TypedDict):
    line: Literal[1, 2]
    code: str
    message: str


class MrzNormalizationReport(TypedDict):
    corrections_count: int
    replacements_count: int
    insertions_count: int
    corrections: list[MrzCorrection]
    issues: list[MrzNormalizationIssue]


def _field_name(line: int, position: int) -> str:
    for start, end, name in MRZ_FIELDS[line]:
        if start <= position <= end:
            return name
    return "За пределами MRZ"


def _correction_reason(
    line: int,
    position: int,
    source: str,
    replacement: str,
    *,
    header: NormalizedPassportHeader | None,
) -> str:
    if not source:
        if line == 1 and position <= 2:
            return "Восстановление пропущенного символа типа документа"
        if line == 2 and position == 1 and replacement == "N":
            return (
                "Восстановление пропущенной "
                "начальной буквы номера паспорта Казахстана"
            )
        if line == 2 and position in (43, 44):
            return "Вычисление отсутствующей контрольной цифры"
        if line == 2:
            return "Дополнение поля дополнительных данных символом <"
        return "Дополнение строки до 44 символов"
    if source.upper() == replacement:
        return "Преобразование в заглавную букву"
    if (
        line == 2
        and position == 1
        and replacement == "N"
        and header is not None
        and header.normalized_header == "P<KAZ"
    ):
        return "Восстановление начальной буквы номера паспорта Казахстана"
    if line == 1 and position <= 2:
        return "Восстановление типа документа"
    if line == 2 and position == 21:
        return "Исправление похожего символа пола"
    if line == 1 and 3 <= position <= 5:
        return "Нормализация кода страны по списку поддерживаемых значений"
    if line == 2 and 11 <= position <= 13:
        return (
            "Нормализация кода гражданства по списку поддерживаемых значений"
        )
    if source in "0123456789" and "A" <= replacement <= "Z":
        return "Замена цифры на похожую букву"
    if source.isalpha() and replacement in "0123456789":
        return "Замена буквы на похожую цифру"
    if replacement == "<":
        return "Замена похожего символа на заполнитель <"
    return "Нормализация символа"


def _inserted_header_position(
    header: NormalizedPassportHeader | None,
) -> int | None:
    if header is None or not header.restored_missing_character:
        return None
    positions = [
        correction.position
        for correction in header.corrections
        if correction.source == "" and correction.replacement
    ]
    if len(positions) != 1 or positions[0] not in (0, 1):
        raise ValueError(
            "A restored document code must identify its inserted character"
        )
    return positions[0]


def build_mrz_normalization_report(
    raw_mrz: Sequence[str],
    normalized_mrz: Sequence[str],
    *,
    header: NormalizedPassportHeader | None,
    lower_line_prefix_inserted: bool = False,
    issues: Sequence[MrzNormalizationIssue] = (),
) -> MrzNormalizationReport:
    """Describe final symbol changes using one-based MRZ field positions.

    Known document-code and number-prefix insertions provide alignment; all
    other insertions are at line ends. Intermediate substitutions at the same
    position collapse into one original-to-final correction.
    """
    inserted_position = _inserted_header_position(header)
    corrections: list[MrzCorrection] = []
    for line, (raw, normalized) in enumerate(
        zip(raw_mrz, normalized_mrz, strict=True), start=1
    ):
        raw_index = 0
        for index, replacement in enumerate(normalized):
            is_inserted_prefix = (
                line == 1 and index == inserted_position
            ) or (line == 2 and index == 0 and lower_line_prefix_inserted)
            if is_inserted_prefix or raw_index >= len(raw):
                source = ""
                source_position = None
            else:
                source = raw[raw_index]
                source_position = raw_index + 1
                raw_index += 1
            if source == replacement:
                continue
            position = index + 1
            corrections.append(
                {
                    "line": line,
                    "position": position,
                    "source_position": source_position,
                    "field": _field_name(line, position),
                    "operation": "замена" if source else "вставка",
                    "source": source,
                    "replacement": replacement,
                    "reason": _correction_reason(
                        line, position, source, replacement, header=header
                    ),
                }
            )
        if raw_index != len(raw):
            raise ValueError(
                "MRZ correction reporting does not support deleted characters"
            )

    replacements = sum(item["operation"] == "замена" for item in corrections)
    return {
        "corrections_count": len(corrections),
        "replacements_count": replacements,
        "insertions_count": len(corrections) - replacements,
        "corrections": corrections,
        "issues": list(issues),
    }

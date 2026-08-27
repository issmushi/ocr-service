from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from app.modules.v1.idcard.profiles import (
    SUPPORTED_IDCARD_COUNTRIES,
    IdcardMrzFormatError,
    IdcardMrzProfile,
    identify_idcard_profile,
    normalize_idcard_country,
    normalize_idcard_header,
)
from app.modules.v1.mrz import calculate_mrz_check_digit
from app.modules.v1.passport.normalization.common import (
    MRZ_DIGIT_TO_LETTER,
    MRZ_LETTER_TO_DIGIT,
    MRZ_SEX_LOOKALIKES,
)


def _compact_lines(lines: Sequence[str]) -> list[str]:
    return [
        compact
        for text in lines
        for line in text.splitlines()
        if (compact := "".join(line.split()).upper())
    ]


def select_longest_idcard_mrz_lines(
    lines: Sequence[str],
    *,
    normalize: bool = True,
) -> tuple[str, ...]:
    """Select the profile's two or three longest rows in OCR source order."""
    compact = _compact_lines(lines)
    profile = None
    for line in compact:
        try:
            header_line = (
                normalize_idcard_header(line)[0] if normalize else line
            )
            profile = identify_idcard_profile(header_line)
            break
        except IdcardMrzFormatError:
            continue
    if profile is None:
        raise IdcardMrzFormatError("Неизвестный формат документа")
    selected = sorted(
        enumerate(compact), key=lambda item: (-len(item[1]), item[0])
    )
    return tuple(line for _, line in sorted(selected[: profile.line_count]))


def _translate_range(
    line: str, start: int, end: int, mapping: dict[int, str]
) -> str:
    return line[:start] + line[start:end].translate(mapping) + line[end:]


def _normalize_country_at(line: str, start: int) -> str:
    if len(line) < start + 3:
        return line
    return (
        line[:start]
        + normalize_idcard_country(line[start : start + 3])
        + line[start + 3 :]
    )


def _normalize_td1_first(line: str, profile: IdcardMrzProfile) -> str:
    if profile.country == "KGZ":
        line = _translate_range(line, 5, 7, MRZ_DIGIT_TO_LETTER)
        line = _translate_range(line, 7, 29, MRZ_LETTER_TO_DIGIT)
        number_pattern = r"[A-Z]{2}[0-9]{8}"
    else:
        line = _translate_range(line, 5, 27, MRZ_LETTER_TO_DIGIT)
        number_pattern = r"[0-9]{10}"
    data_end = 15 + profile.personal_number_length
    if (
        data_end <= len(line) < 30
        and re.fullmatch(number_pattern, line[5:15]) is not None
        and line[15:data_end].isdigit()
        and not line[data_end:].strip("<")
    ):
        line = line.ljust(30, "<")
    return line


def _normalize_td1_second(
    line: str, first: str, profile: IdcardMrzProfile
) -> str:
    for start, end in ((0, 7), (8, 15), (29, 30)):
        line = _translate_range(line, start, end, MRZ_LETTER_TO_DIGIT)
    line = _translate_range(line, 7, 8, MRZ_SEX_LOOKALIKES)
    line = _normalize_country_at(line, 15)
    personal_end = 15 + profile.personal_number_length
    number_pattern = (
        r"[A-Z]{2}[0-9]{8}" if profile.country == "KGZ" else r"[0-9]{10}"
    )
    if (
        18 <= len(line) < 30
        and len(first) == 30
        and re.fullmatch(number_pattern, first[5:15]) is not None
        and first[15:personal_end].isdigit()
        and not first[personal_end:].strip("<")
        and line[:7].isdigit()
        and line[8:15].isdigit()
        and line[7] in {"M", "F", "<"}
        and line[15:18] in SUPPORTED_IDCARD_COUNTRIES
        and not line[18:].strip("<")
    ):
        line = line.ljust(29, "<")
        composite = first[5:30] + line[:7] + line[8:15] + line[18:29]
        line += calculate_mrz_check_digit(composite)
    return line


def _normalize_td2_second(line: str) -> str:
    for start, end in ((0, 10), (13, 20), (21, 34), (35, 36)):
        line = _translate_range(line, start, end, MRZ_LETTER_TO_DIGIT)
    line = _normalize_country_at(line, 10)
    line = _translate_range(line, 20, 21, MRZ_SEX_LOOKALIKES)
    if (
        34 <= len(line) < 36
        and line[:10].isdigit()
        and line[10:13] in SUPPORTED_IDCARD_COUNTRIES
        and line[13:20].isdigit()
        and line[20] in {"M", "F", "<"}
        and line[21:34].isdigit()
        and not line[34:].strip("<")
    ):
        line = line.ljust(35, "<")
        line += calculate_mrz_check_digit(
            line[:10] + line[13:20] + line[21:35]
        )
    return line


def _field_name(profile: IdcardMrzProfile, line: int, position: int) -> str:
    if line == 1 and position <= 5:
        return "Тип документа" if position <= 2 else "Страна выдачи"
    if profile.format == "TD2":
        if line == 1:
            return "Фамилия и имена" if position <= 36 else "За пределами MRZ"
        ranges = (
            (1, 9, "Номер документа"),
            (10, 10, "Контрольная цифра номера документа"),
            (11, 13, "Гражданство"),
            (14, 19, "Дата рождения"),
            (20, 20, "Контрольная цифра даты рождения"),
            (21, 21, "Пол"),
            (22, 27, "Дата окончания срока действия"),
            (28, 28, "Контрольная цифра срока действия"),
            (29, 34, "ИИН"),
            (35, 35, "Дополнительные данные"),
            (36, 36, "Общая контрольная цифра"),
        )
    elif line == 3:
        return "Фамилия и имена" if position <= 30 else "За пределами MRZ"
    elif line == 1:
        personal_end = 15 + profile.personal_number_length
        ranges = (
            (6, 14, "Номер документа"),
            (15, 15, "Контрольная цифра номера документа"),
            (16, personal_end, "ИИН" if profile.country == "KAZ" else "ПИН"),
            (personal_end + 1, 30, "Дополнительные данные"),
        )
    else:
        ranges = (
            (1, 6, "Дата рождения"),
            (7, 7, "Контрольная цифра даты рождения"),
            (8, 8, "Пол"),
            (9, 14, "Дата окончания срока действия"),
            (15, 15, "Контрольная цифра срока действия"),
            (16, 18, "Гражданство"),
            (19, 29, "Дополнительные данные"),
            (30, 30, "Общая контрольная цифра"),
        )
    return next(
        (name for start, end, name in ranges if start <= position <= end),
        "За пределами MRZ",
    )


def _reason(
    source: str, replacement: str, field: str, line_length: int
) -> str:
    if not source:
        if field == "Тип документа":
            return "Восстановление пропущенного символа типа документа"
        if field == "Общая контрольная цифра":
            return "Вычисление отсутствующей контрольной цифры"
        if field == "Фамилия и имена":
            return f"Дополнение строки до {line_length} символов"
        return "Дополнение поля дополнительных данных символом <"
    if field == "Тип документа":
        return "Восстановление типа документа"
    if field in {"Страна выдачи", "Гражданство"}:
        return "Нормализация кода страны по списку поддерживаемых значений"
    if field == "Пол":
        return "Исправление похожего символа пола"
    if source.isdigit() and replacement.isalpha():
        return "Замена цифры на похожую букву"
    if source.isalpha() and replacement.isdigit():
        return "Замена буквы на похожую цифру"
    return "Нормализация символа"


def _report(
    raw: Sequence[str],
    normalized: Sequence[str],
    profile: IdcardMrzProfile,
    inserted_header_position: int | None,
) -> dict[str, Any]:
    corrections = []
    for line_number, (before, after) in enumerate(
        zip(raw, normalized, strict=True), 1
    ):
        source_index = 0
        for index, replacement in enumerate(after):
            inserted = line_number == 1 and index == inserted_header_position
            source = (
                ""
                if inserted or source_index >= len(before)
                else before[source_index]
            )
            source_position = source_index + 1 if source else None
            source_index += bool(source)
            if source == replacement:
                continue
            field = _field_name(profile, line_number, index + 1)
            corrections.append(
                {
                    "line": line_number,
                    "position": index + 1,
                    "source_position": source_position,
                    "field": field,
                    "operation": "замена" if source else "вставка",
                    "source": source,
                    "replacement": replacement,
                    "reason": _reason(
                        source, replacement, field, profile.line_length
                    ),
                }
            )
        if source_index != len(before):
            raise ValueError(
                "ID-card normalization must preserve every source character"
            )
    replacements = sum(item["operation"] == "замена" for item in corrections)
    return {
        "corrections_count": len(corrections),
        "replacements_count": replacements,
        "insertions_count": len(corrections) - replacements,
        "corrections": corrections,
        "issues": [],
    }


def normalize_idcard_mrz(
    lines: Sequence[str], *, normalize: bool = True
) -> dict[str, Any]:
    """Apply passport-style field corrections through one
    explicit ID profile.
    """
    raw = list(select_longest_idcard_mrz_lines(lines, normalize=normalize))
    normalized = list(raw)
    inserted_header_position = None
    if normalize:
        normalized[0], inserted_header_position = normalize_idcard_header(
            normalized[0]
        )
    profile = identify_idcard_profile(normalized[0])
    if len(normalized) != profile.line_count:
        raise IdcardMrzFormatError("Неполная MRZ документа")
    if normalize:
        if profile.format == "TD1":
            normalized[0] = _normalize_td1_first(normalized[0], profile)
            if len(normalized) > 1:
                normalized[1] = _normalize_td1_second(
                    normalized[1], normalized[0], profile
                )
            if len(normalized) > 2:
                normalized[2] = (
                    normalized[2].translate(MRZ_DIGIT_TO_LETTER).ljust(30, "<")
                )
        else:
            normalized[0] = (
                normalized[0].translate(MRZ_DIGIT_TO_LETTER).ljust(36, "<")
            )
            if len(normalized) > 1:
                normalized[1] = _normalize_td2_second(normalized[1])
    return {
        "mrz": normalized,
        "raw_mrz": raw,
        "diagnostics": {
            "format": profile.format,
            "issuing_country": profile.country,
            "normalization": _report(
                raw, normalized, profile, inserted_header_position
            ),
        },
    }

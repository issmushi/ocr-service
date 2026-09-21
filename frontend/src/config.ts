import type { ParamDefinition, ParamValue, ToolDefinition, ToolId } from "./types";

const mrzModels = [
  { value: "paddle", label: "PaddleOCR", note: "Чтение текста" },
  { value: "mrz", label: "MRZ Scanner", note: "Чтение MRZ" },
  { value: "qwen3-vl-4b", label: "Qwen3-VL 4B", note: "Если включён на сервере" },
  { value: "qwen3-vl-8b", label: "Qwen3-VL 8B", note: "Если включён на сервере" },
];

const mrzParams: ParamDefinition[] = [
  {
    kind: "select",
    key: "model",
    label: "Модель",
    description: "Движок для чтения MRZ.",
    defaultValue: "paddle",
    options: mrzModels,
  },
  {
    kind: "toggle",
    key: "crop",
    label: "Найти MRZ",
    description: "Обрезать документ и выделить MRZ перед распознаванием.",
    defaultValue: true,
  },
  {
    kind: "toggle",
    key: "normalize",
    label: "Нормализовать",
    description: "Применить замены символов и дополнение MRZ по правилам страны.",
    defaultValue: true,
  },
  {
    kind: "toggle",
    key: "normalization",
    label: "Показать журнал",
    description: "Добавить в ответ список исправлений. Обработка зависит от normalize.",
    defaultValue: true,
  },
];

export const tools: ToolDefinition[] = [
  {
    id: "passport-mrz",
    group: "Распознавание",
    title: "MRZ паспорта",
    shortTitle: "Passport MRZ",
    description: "Прочитать две строки MRZ и разобрать их в поля паспорта.",
    endpoint: "/v1/ocr/passport/mrz",
    responseKind: "json",
    params: mrzParams,
    outputHint: "Ответ: MRZ, поля паспорта, normalization и meta",
  },
  {
    id: "passport-page",
    group: "Распознавание",
    title: "Страница паспорта",
    shortTitle: "Passport page",
    description: "Прочитать ФИО и другие поля на странице с фото в паспорте РФ.",
    endpoint: "/v1/ocr/passport/page",
    responseKind: "json",
    params: [
      {
        kind: "select",
        key: "page",
        label: "Страница",
        description: "Пока только страница 3 — с фото и ФИО.",
        defaultValue: "3",
        options: [{ value: "3", label: "Страница 3" }],
      },
      {
        kind: "select",
        key: "model",
        label: "Модель",
        description: "Модель для чтения текста на странице.",
        defaultValue: "russian-ocr",
        options: [
          { value: "russian-ocr", label: "Russian OCR", note: "Поля паспорта РФ" },
          { value: "qwen3-vl-4b", label: "Qwen3-VL 4B", note: "Если включён на сервере" },
          { value: "qwen3-vl-8b", label: "Qwen3-VL 8B", note: "Если включён на сервере" },
        ],
      },
      {
        kind: "toggle",
        key: "crop",
        label: "Найти документ",
        description: "Выровнять и обрезать страницу перед OCR.",
        defaultValue: true,
      },
    ],
    outputHint: "Ответ: ФИО, даты, серия, номер и meta",
  },
  {
    id: "idcard-mrz",
    group: "Распознавание",
    title: "MRZ ID-карты",
    shortTitle: "ID-card MRZ",
    description: "Прочитать MRZ ID-карты Казахстана или Кыргызстана и разобрать поля.",
    endpoint: "/v1/ocr/idcard/mrz",
    responseKind: "json",
    params: mrzParams,
    outputHint: "Ответ: строки MRZ, поля ID-карты, normalization и meta",
  },
  {
    id: "full-ocr",
    group: "Распознавание",
    title: "Весь текст",
    shortTitle: "Full OCR",
    description: "Вернуть найденный текст без разбора на поля документа.",
    endpoint: "/v1/ocr/full",
    responseKind: "json",
    params: [
      {
        kind: "select",
        key: "model",
        label: "Модель",
        description: "Модель для чтения текста на фото.",
        defaultValue: "paddle",
        options: [
          { value: "paddle", label: "PaddleOCR", note: "Чтение текста" },
          { value: "qwen3-vl-4b", label: "Qwen3-VL 4B", note: "Если включён на сервере" },
          { value: "qwen3-vl-8b", label: "Qwen3-VL 8B", note: "Если включён на сервере" },
        ],
      },
    ],
    outputHint: "Ответ: text и meta",
  },
  {
    id: "crop-document",
    group: "Обрезка и качество",
    title: "Обрезка документа",
    shortTitle: "Crop document",
    description: "Найти документ на фото, выровнять и обрезать по границам.",
    endpoint: "/v1/ocr/crop/document",
    responseKind: "image",
    params: [],
    outputHint: "Ответ: обрезанный документ в JPEG",
  },
  {
    id: "crop-mrz",
    group: "Обрезка и качество",
    title: "Обрезка MRZ",
    shortTitle: "Crop MRZ",
    description: "Найти MRZ на фото и вернуть её отдельным изображением.",
    endpoint: "/v1/ocr/crop/mrz",
    responseKind: "image",
    params: [],
    outputHint: "Ответ: обрезанная MRZ в JPEG",
  },
  {
    id: "best-frame",
    group: "Обрезка и качество",
    title: "Лучший кадр",
    shortTitle: "Best frame",
    description: "Сравнить до трёх фото одного документа по резкости, бликам и другим признакам.",
    endpoint: "/v1/ocr/best-frame",
    responseKind: "image",
    multiple: true,
    maxFiles: 3,
    params: [],
    outputHint: "Ответ: выбранное фото, индекс и оценка в заголовках",
  },
];

export const toolById = Object.fromEntries(
  tools.map((tool) => [tool.id, tool]),
) as Record<ToolId, ToolDefinition>;

export function defaultParams(tool: ToolDefinition): Record<string, ParamValue> {
  return Object.fromEntries(tool.params.map((param) => [param.key, param.defaultValue]));
}

export const acceptedFormats = ".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff";

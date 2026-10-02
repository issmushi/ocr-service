import type { ParamDefinition, ParamValue, ToolDefinition, ToolId } from "./types";

const passportFieldModels = [
  { value: "none", label: "Только вырезки" },
  { value: "paddle", label: "Paddle eslav v5 · CPU" },
  { value: "paddle-cyrillic", label: "Paddle Cyrillic v5 · CPU" },
  { value: "russian-ocr-accurate", label: "Russian OCR accurate · CPU" },
  { value: "qwen3-vl-4b", label: "Qwen3-VL 4B", note: "Если включён на сервере" },
  { value: "qwen3-vl-8b", label: "Qwen3-VL 8B", note: "Если включён на сервере" },
];

export function passportFieldOcrLabel(model: string): string {
  return passportFieldModels.find((option) => option.value === model)?.label ?? model;
}

const mrzParams: ParamDefinition[] = [
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
    id: "mrz",
    group: "MRZ",
    title: "MRZ документа",
    shortTitle: "Document MRZ",
    description: "С помощью MRZ Scanner прочитать две или три строки и разобрать поля паспорта или ID-карты.",
    endpoint: "/v1/ocr/mrz",
    responseKind: "json",
    params: mrzParams,
    outputHint: "Ответ: MRZ, поля документа, normalization и meta",
  },
  {
    id: "passport-internal",
    group: "Распознавание",
    title: "Внутренний паспорт РФ",
    shortTitle: "Internal passport",
    description: "Распознать поля внутреннего паспорта РФ на фото с помощью Russian OCR.",
    endpoint: "/v1/ocr/passport/internal",
    responseKind: "json",
    params: [],
    outputHint: "Ответ: поля паспорта, MRZ при наличии и meta",
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
    id: "crop-passport-fields",
    group: "Обрезка и качество",
    title: "Поля паспорта",
    shortTitle: "Passport fields",
    description: "Найти поля паспорта РФ и сравнить чтение вырезок моделями Paddle eslav v5, Paddle Cyrillic v5, Russian OCR accurate и Qwen. Текст возвращается без нормализации.",
    endpoint: "/v1/ocr/crop/passport-fields",
    responseKind: "json",
    params: [
      {
        kind: "select",
        key: "ocr_model",
        label: "Распознавание полей",
        description: "Paddle eslav v5, Paddle Cyrillic v5 и Russian OCR accurate работают на CPU; Qwen — в режиме сервера. Каждое поле читается отдельно, фото и подпись пропускаются.",
        defaultValue: "paddle",
        options: passportFieldModels,
      },
      {
        kind: "toggle",
        key: "crop_document",
        label: "Найти документ",
        description: "Выровнять и обрезать документ перед поиском полей.",
        defaultValue: true,
      },
      {
        kind: "select",
        key: "padding",
        label: "Отступ вырезки",
        description: "Добавить пиксели вокруг найденного поля.",
        defaultValue: "4",
        options: [0, 4, 8, 16, 32, 64].map((value) => ({ value: String(value), label: `${value} px` })),
      },
    ],
    outputHint: "Ответ: вырезки полей, сырой текст выбранной модели и время обработки",
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

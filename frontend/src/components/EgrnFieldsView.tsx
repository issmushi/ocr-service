import { useMemo } from "react";
import { passportFieldOcrLabel } from "../config";
import { FieldOcrView } from "./FieldOcrView";
import { isFieldOcr, isOcrSummary, type PassportFieldOcr, type FieldOcrSummary } from "./passportFields";

const fieldLabels = {
  application_number: "Номер заявления",
  object_type: "Вид объекта",
  cadastral_number: "Кадастровый номер",
  cadastral_quarter_number: "Номер кадастрового квартала",
  cadastral_number_assignment_date: "Дата присвоения кадастрового номера",
  previous_registration_number: "Ранее присвоенный учётный номер",
  address: "Адрес (местоположение)",
  floor: "Номер, тип этажа",
  area: "Площадь, м²",
  purpose: "Назначение",
  object_name: "Наименование объекта",
  residential_premises_type: "Вид жилого помещения",
  cadastral_value: "Кадастровая стоимость, руб.",
  parent_cadastral_numbers: "Кадастровые номера содержащих объектов",
  cadastral_engineer: "Сведения о кадастровом инженере",
  cultural_heritage: "Сведения о включении в реестр культурного наследия",
  claims: "Сведения о правопритязаниях",
  specialized_housing: "Отнесение к специализированному жилищному фонду",
  actuality_date: "Дата актуальности",
} as const;

type FieldName = keyof typeof fieldLabels;
interface Preview { image: string; width: number; height: number }
interface Field extends Preview { ocr?: PassportFieldOcr | null; name: FieldName; label: string; bbox: [number, number, number, number] }
interface Result { document: Preview; fields: Field[]; missing_fields: FieldName[]; meta: { duration_seconds: number; ocr?: FieldOcrSummary | null } }

function record(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}
function preview(value: unknown): value is Preview {
  return record(value) && typeof value.image === "string"
    && /^data:image\/png;base64,[A-Za-z0-9+/]+={0,2}$/.test(value.image)
    && Number.isSafeInteger(value.width) && Number(value.width) > 0
    && Number.isSafeInteger(value.height) && Number(value.height) > 0;
}
function fieldName(value: unknown): value is FieldName {
  return typeof value === "string" && Object.hasOwn(fieldLabels, value);
}

export function parseEgrnFields(value: unknown): Result | null {
  if (!record(value) || !preview(value.document) || !Array.isArray(value.fields)
    || !Array.isArray(value.missing_fields) || !record(value.meta)
    || typeof value.meta.duration_seconds !== "number" || !Number.isFinite(value.meta.duration_seconds)
    || value.meta.duration_seconds < 0 || (value.meta.ocr != null && !isOcrSummary(value.meta.ocr))) return null;
  const names = new Set<string>();
  for (const field of value.fields) {
    if (!record(field) || !preview(field) || !fieldName(field.name)
      || (field.ocr != null && !isFieldOcr(field.ocr))
      || names.has(field.name) || typeof field.label !== "string" || !field.label.trim()
      || !Array.isArray(field.bbox) || field.bbox.length !== 4
      || !field.bbox.every((coordinate) => Number.isSafeInteger(coordinate) && coordinate >= 0)) return null;
    const [x1, y1, x2, y2] = field.bbox;
    if (x2 <= x1 || y2 <= y1 || x2 > value.document.width || y2 > value.document.height
      || field.width !== x2 - x1 || field.height !== y2 - y1) return null;
    names.add(field.name);
  }
  for (const name of value.missing_fields) {
    if (!fieldName(name) || names.has(name)) return null;
    names.add(name);
  }
  return names.size === Object.keys(fieldLabels).length ? value as unknown as Result : null;
}

export function EgrnFieldsView({ data }: { data: unknown }) {
  const result = useMemo(() => parseEgrnFields(data), [data]);
  if (!result) return <div className="tab-empty" role="alert">Не удалось показать вырезки ЕГРН: неожиданный формат ответа. Полный ответ доступен во вкладке JSON.</div>;
  const { document, fields, missing_fields: missing, meta } = result;
  const badge = Math.min(document.width, document.height) / 35;
  return <div className="passport-fields">
    <div className="passport-fields__summary"><strong>Найдено участков: {fields.length} из {Object.keys(fieldLabels).length}</strong><span>Обработка: {meta.duration_seconds.toFixed(2)} с</span></div>
    <p className="passport-fields__note">Первая страница выписки Госуслуг. Рамки показывают точные границы вырезок на выровненной странице. {meta.ocr ? "Каждая вырезка распознаётся отдельно, без нормализации." : "Распознавание текста выключено."}</p>
    {meta.ocr && <div className="passport-fields__ocr-summary"><span>OCR: {passportFieldOcrLabel(meta.ocr.model)} · {meta.ocr.duration_seconds.toFixed(2)} с</span><span>Обработано полей: {meta.ocr.processed_fields} · Ошибок: {meta.ocr.failed_fields}</span></div>}
    {missing.length > 0 && <div className="passport-fields__empty" role="status"><strong>Не найдены:</strong><p>{missing.map((name) => fieldLabels[name]).join(", ")}</p></div>}
    <figure className="passport-fields__document">
      <svg aria-label={`Выписка ЕГРН: ${fields.length} выделенных участков`} height={document.height} role="img" viewBox={`0 0 ${document.width} ${document.height}`} width={document.width}>
        <image height={document.height} href={document.image} width={document.width} />
        {fields.map((field, index) => {
          const [x1, y1, x2, y2] = field.bbox;
          const labelY = Math.max(0, y1 - badge);
          return <g key={field.name}><title>{`${index + 1}. ${field.label}`}</title>
            <rect className="passport-fields__box" height={y2 - y1} vectorEffect="non-scaling-stroke" width={x2 - x1} x={x1} y={y1} />
            <rect className="passport-fields__badge" height={badge} rx={badge / 7} width={badge} x={x1} y={labelY} />
            <text dominantBaseline="central" fontSize={badge * 0.67} textAnchor="middle" x={x1 + badge / 2} y={labelY + badge / 2}>{index + 1}</text>
          </g>;
        })}
      </svg>
      <figcaption><span>{document.width} × {document.height} px</span><a download="egrn-document.png" href={document.image}>Скачать страницу PNG</a></figcaption>
    </figure>
    <div className="passport-fields__grid">
      {fields.map((field, index) => <article className="passport-field" key={field.name}>
        <header><span>{index + 1}</span><div><h3>{field.label}</h3><code>{field.name}</code></div></header>
        <div className="passport-field__image"><img alt={`Вырезка: ${field.label}`} height={field.height} src={field.image} width={field.width} /></div>
        <dl><div><dt>Размер</dt><dd>{field.width} × {field.height} px</dd></div><div><dt>Координаты</dt><dd><code>{field.bbox.join(", ")}</code></dd></div></dl>
        {field.ocr && <FieldOcrView ocr={field.ocr} />}
        <a className="button button--secondary" download={`egrn-${field.name}.png`} href={field.image}>Скачать PNG</a>
      </article>)}
    </div>
  </div>;
}

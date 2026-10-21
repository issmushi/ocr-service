import { useMemo } from "react";
import { passportFieldOcrLabel } from "../config";
import { parsePassportFieldCrops, type PassportFieldOcr } from "./passportFields";

function FieldOcrView({ ocr }: { ocr: PassportFieldOcr }) {
  return <section className={`passport-field__ocr passport-field__ocr--${ocr.status}`} aria-label="Результат OCR">
    <strong>Распознавание</strong>
    {ocr.status === "success" && (ocr.text?.trim()
      ? <pre className="passport-field__text">{ocr.text}</pre>
      : <p className="passport-field__ocr-message">Текст не найден</p>)}
    {ocr.status === "error" && <p className="passport-field__ocr-message" role="status">Ошибка: {ocr.error?.message || "Не удалось распознать поле"}{ocr.error?.code && <code>{ocr.error.code}</code>}</p>}
    {ocr.status === "skipped" && <p className="passport-field__ocr-message">Поле без печатного текста</p>}
    <dl>
      <div><dt>Модель OCR</dt><dd>{passportFieldOcrLabel(ocr.model)}</dd></div>
      <div><dt>Движок OCR</dt><dd>{ocr.engine}</dd></div>
      <div><dt>Устройство OCR</dt><dd>{ocr.device?.toUpperCase() || "Не указано"}</dd></div>
      <div><dt>Время OCR</dt><dd>{ocr.duration_seconds.toFixed(2)} с</dd></div>
      {ocr.confidence !== null && <div><dt>Уверенность OCR</dt><dd>{(ocr.confidence * 100).toFixed(1)}%</dd></div>}
    </dl>
  </section>;
}

export function PassportFieldsView({ data }: { data: unknown }) {
  const result = useMemo(() => parsePassportFieldCrops(data), [data]);
  if (!result) return <div className="tab-empty" role="alert">Не удалось показать вырезки: ответ API имеет неожиданный формат. Полный ответ доступен во вкладке JSON.</div>;
  const { document, fields, durationSeconds, ocr } = result;
  const badgeSize = Math.min(document.width, document.height) / 22;

  return <div className="passport-fields">
    <div className="passport-fields__summary">
      <strong>Найдено полей: {fields.length}</strong>
      {durationSeconds !== null && <span>Обработка: {durationSeconds.toFixed(2)} с</span>}
    </div>
    {ocr && <div className="passport-fields__ocr-summary">
      <span>OCR: {passportFieldOcrLabel(ocr.model)} · {ocr.duration_seconds.toFixed(2)} с</span>
      <span>Обработано полей: {ocr.processed_fields} · Ошибок: {ocr.failed_fields}</span>
    </div>}
    <figure className="passport-fields__document">
      <svg aria-label={`Обработанный документ: ${fields.length} областей полей`} height={document.height} role="img" viewBox={`0 0 ${document.width} ${document.height}`} width={document.width}>
        <image height={document.height} href={document.image} width={document.width} />
        {fields.map((field, index) => {
          const [x1, y1, x2, y2] = field.detection_bbox;
          const badgeX = Math.min(x1, document.width - badgeSize);
          const badgeY = Math.min(Math.max(0, y1 - badgeSize), document.height - badgeSize);
          return <g key={`${field.name}-${index}`}>
            <title>{`${index + 1}. ${field.label} · ${(field.confidence * 100).toFixed(1)}%`}</title>
            <rect className="passport-fields__box" height={y2 - y1} vectorEffect="non-scaling-stroke" width={x2 - x1} x={x1} y={y1} />
            <rect className="passport-fields__badge" height={badgeSize} rx={badgeSize / 7} width={badgeSize} x={badgeX} y={badgeY} />
            <text dominantBaseline="central" fontSize={badgeSize * 0.67} textAnchor="middle" x={badgeX + badgeSize / 2} y={badgeY + badgeSize / 2}>{index + 1}</text>
          </g>;
        })}
      </svg>
      <figcaption>
        <span>{document.width} × {document.height} px · {document.detected ? "Документ найден" : "Без обрезки по границам документа"}</span>
        <span>Поворот документа: {document.rotation_degrees}°{document.orientation_confidence !== null && ` · Уверенность ориентации: ${(document.orientation_confidence * 100).toFixed(1)}%`}</span>
      </figcaption>
    </figure>
    <p className="passport-fields__note">Рамки показывают области детектора на документе. Вырезки включают выбранный отступ. {ocr ? "Каждое текстовое поле распознаётся отдельно, без нормализации." : "Распознавание текста выключено."}</p>
    {fields.length === 0 ? <div className="passport-fields__empty" role="status"><strong>Поля не найдены</strong><p>Попробуйте другой снимок или измените параметр «Найти документ».</p></div> : <div className="passport-fields__grid">
      {fields.map((field, index) => <article className="passport-field" key={`${field.name}-${index}`}>
        <header><span>{index + 1}</span><div><h3>{field.label}</h3><code>{field.name}</code></div></header>
        <div className="passport-field__image"><img alt={`Вырезка: ${field.label}`} height={field.height} src={field.image} width={field.width} /></div>
        <dl>
          <div><dt>Размер</dt><dd>{field.width} × {field.height} px</dd></div>
          <div><dt>Уверенность детектора</dt><dd>{(field.confidence * 100).toFixed(1)}%</dd></div>
          {field.rotation_degrees !== 0 && <div><dt>Поворот вырезки</dt><dd>{field.rotation_degrees}° против часовой стрелки</dd></div>}
          <div><dt>Область вырезки</dt><dd><code>{field.bbox.join(", ")}</code></dd></div>
        </dl>
        {field.ocr && <FieldOcrView ocr={field.ocr} />}
      </article>)}
    </div>}
  </div>;
}

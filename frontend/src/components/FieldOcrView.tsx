import { passportFieldOcrLabel } from "../config";
import type { FieldOcr } from "./fieldOcr";

export function FieldOcrView({ ocr }: { ocr: FieldOcr }) {
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

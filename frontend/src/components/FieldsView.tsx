const fieldLabels: Record<string, string> = {
  passport_type: "Тип паспорта",
  document_type: "Тип документа",
  document_code: "Код документа",
  issuing_country: "Страна выдачи",
  surname: "Фамилия",
  name: "Имя",
  patronymic: "Отчество",
  sex: "Пол",
  date_of_birth: "Дата рождения",
  passport_series: "Серия паспорта",
  passport_number: "Номер паспорта",
  document_series: "Серия документа",
  document_number: "Номер документа",
  nationality: "Гражданство",
  issue_date: "Дата выдачи",
  Issue_organization_ru: "Кем выдан",
  expiry_date: "Срок действия",
  department_code: "Код подразделения",
  personal_number: "Персональный номер",
  mrz_valid: "MRZ валидна",
  text: "Распознанный текст",
};

const documentTypeLabels: Record<string, string> = {
  internal_passport: "Внутренний паспорт",
  international_passport: "Заграничный паспорт",
  identity_card: "ID-карта",
};

export function FieldsView({ entries }: { entries: [string, unknown][] }) {
  if (!entries.length) return <div className="tab-empty">В ответе нет полей документа.</div>;
  return <div className="fields-grid">{entries.map(([key, value]) => (
    key === "text" ? (
      <div className="field-item field-item--text" key={key}>
        <span>{fieldLabels[key]}</span>
        <pre>{String(value ?? "")}</pre>
      </div>
    ) :
    <div className={key === "mrz_valid" ? "field-item field-item--highlight" : "field-item"} key={key}>
      <span>{fieldLabels[key] ?? key.replaceAll("_", " ")}</span>
      <strong className={value === null || value === "" ? "muted" : ""}>{value === null || value === "" ? "Не найдено" : typeof value === "boolean" ? value ? "Да" : "Нет" : key === "document_type" && typeof value === "string" ? documentTypeLabels[value] ?? value : String(value)}</strong>
    </div>
  ))}</div>;
}

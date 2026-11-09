const fieldLabels: Record<string, string> = {
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
  passport_type: "Тип паспорта",
  document_type: "Тип документа",
  document_code: "Код документа",
  issuing_country: "Страна выдачи",
  surname: "Фамилия",
  surname_latin: "Фамилия (латиницей)",
  name: "Имя",
  name_latin: "Имя (латиницей)",
  patronymic: "Отчество",
  patronymic_latin: "Отчество (латиницей)",
  sex: "Пол",
  date_of_birth: "Дата рождения",
  birth_place_ru: "Место рождения (RU)",
  birth_place_en: "Место рождения (EN)",
  passport_series: "Серия паспорта",
  passport_number: "Номер паспорта",
  snils_number: "Номер СНИЛС",
  license_series: "Серия удостоверения",
  license_number: "Номер удостоверения",
  birth_place: "Место рождения",
  registration_date: "Дата регистрации",
  issue_organization_ru: "Кем выдан",
  issue_organization_en: "Кем выдан (латиницей)",
  residence_region_ru: "Регион проживания",
  residence_region_en: "Регион проживания (латиницей)",
  categories: "Категории",
  document_series: "Серия документа",
  document_number: "Номер документа",
  nationality: "Гражданство",
  issue_date: "Дата выдачи",
  Issue_organization_ru: "Кем выдан",
  issue_organization_code: "Код органа выдачи",
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

function isMissing(value: unknown): boolean {
  return value === null || value === "" || Array.isArray(value) && value.length === 0;
}

function formatFieldValue(key: string, value: unknown): string {
  if (isMissing(value)) return "Не найдено";
  if (typeof value === "boolean") return value ? "Да" : "Нет";
  if (key === "document_type" && typeof value === "string") return documentTypeLabels[value] ?? value;
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

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
      <strong className={isMissing(value) ? "muted" : ""}>{formatFieldValue(key, value)}</strong>
    </div>
  ))}</div>;
}

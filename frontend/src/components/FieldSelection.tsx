import type { MultiselectParam } from "../types";

export function toggleFieldSelection(current: string[], field: string): string[] {
  if (field === "all") return current.includes("all") ? [] : ["all"];
  const selected = current.filter((value) => value !== "all");
  return selected.includes(field) ? selected.filter((value) => value !== field) : [...selected, field];
}

export function FieldSelection({ param, value, onChange }: {
  param: MultiselectParam;
  value: string[];
  onChange: (value: string[]) => void;
}) {
  return <fieldset className="field-selection">
    <legend>{param.label}</legend>
    <p>{param.description}</p>
    <div className="field-selection__options">
      {param.options.map((option) => <label key={option.value}>
        <input type="checkbox" checked={value.includes(option.value)} onChange={() => onChange(toggleFieldSelection(value, option.value))} />
        <span>{option.label}</span>
      </label>)}
    </div>
    {!value.length && <p role="alert">Выберите хотя бы одно поле или «Все поля».</p>}
  </fieldset>;
}

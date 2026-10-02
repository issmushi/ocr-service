import { useEffect, useState } from "react";
import { Icon } from "../components/Icon";
import { tools } from "../config";
import type { Section, ToolId } from "../types";

interface Props {
  navigate: (section: Section, tool?: ToolId) => void;
}

const docSections = [
  ["quickstart", "Быстрый старт"],
  ["endpoints", "Эндпоинты"],
  ["tasks", "Задачи"],
  ["pipeline", "Обработка MRZ"],
  ["normalization", "Нормализация"],
  ["documents", "Документы"],
  ["limits", "Форматы и лимиты"],
  ["errors", "Ошибки"],
] as const;

const countryRules = [
  {
    code: "ARM",
    name: "Армения",
    header: "P<ARM",
    rules: ["цифры в строке ФИО → похожие буквы", "первые две позиции номера сохраняются буквенными", "числовые поля и контрольные позиции → цифры", "пол N/H → M, E → F", "пустой хвост дополняется заполнителями и контрольными цифрами"],
  },
  {
    code: "KAZ",
    name: "Казахстан",
    header: "P<KAZ",
    rules: ["восстанавливается пропущенная N перед номером", "даты, номер, ИИН и контрольные позиции → цифры", "код гражданства сверяется со списком поддерживаемых кодов", "пол N/H → M, E → F", "хвост достраивается только при полном ИИН"],
  },
  {
    code: "RUS",
    name: "Россия · заграничный",
    header: "P<RUS",
    rules: ["буквы в числовых полях → похожие цифры", "код гражданства и пол исправляются по позиции в строке", "пустое дополнительное поле дополняется символами <", "если обязательные поля прошли проверку, достраивается хвост и контрольные позиции 43–44"],
  },
  {
    code: "RUS",
    name: "Россия · внутренний",
    header: "PNRUS",
    rules: ["в ФИО исправляются только 0→O, 1→I, 5→S", "номер, дата рождения, дата выдачи и код подразделения → цифры", "серию и номер извлекает отдельный парсер", "хвост достраивается при наличии обязательных данных"],
  },
  {
    code: "TJK",
    name: "Таджикистан",
    header: "P<TJK / PPTJK",
    rules: ["поддерживаются оба вида заголовка", "символы в числовых полях → цифры", "код гражданства и пол исправляются по позиции в строке", "13 цифр дополнительных данных нормализуются и не заменяются заполнителями", "контрольный хвост достраивается при наличии всех 13 цифр"],
  },
  {
    code: "UZB",
    name: "Узбекистан",
    header: "P<UZB",
    rules: ["первые две позиции номера сохраняются буквенными", "оставшаяся часть номера и даты → цифры", "код гражданства и пол исправляются по позиции в строке", "14 цифр дополнительных данных проверяются до достройки хвоста"],
  },
];

const errors = [
  ["400", "NO_IMAGE / EMPTY_IMAGE", "Файл отсутствует или пуст"],
  ["400", "UNSUPPORTED_FORMAT", "Расширение файла не поддерживается"],
  ["400", "IMAGE_READ_ERROR", "Изображение не удалось декодировать"],
  ["404", "TASK_NOT_FOUND", "Задача с таким UUID не найдена"],
  ["409", "TASK_NOT_READY / TASK_CANCELLED", "Задача ещё не завершена или отменена"],
  ["413", "IMAGE_TOO_LARGE", "Превышен лимит размера файла или изображения"],
  ["422", "VALIDATION_ERROR", "Параметр не прошёл проверку"],
  ["422", "OCR_FAILED", "Движок не получил пригодный результат"],
  ["429", "QUEUE_FULL", "Очередь асинхронных задач заполнена"],
  ["500", "INTERNAL_ERROR", "Внутренняя ошибка сервиса"],
  ["503", "OCR_ENGINE_UNAVAILABLE", "Модель отключена или недоступна"],
  ["504", "OCR_TIMEOUT", "Превышен таймаут обработки"],
];

const taskOperations = [
  ["/v1/ocr/mrz", "/v1/tasks/mrz"],
  ["/v1/ocr/passport/internal", "/v1/tasks/passport/internal"],
  ["/v1/ocr/full", "/v1/tasks/full"],
  ["/v1/ocr/crop/document", "/v1/tasks/crop/document"],
  ["/v1/ocr/crop/mrz", "/v1/tasks/crop/mrz"],
] as const;

export function DocsPage({ navigate }: Props) {
  const [active, setActive] = useState("quickstart");
  const [copied, setCopied] = useState(false);
  const curl = `BASE_URL=http://127.0.0.1:8000\ncurl -X POST "$BASE_URL/v1/ocr/mrz" \\\n  -F 'image=@passport.jpg' \\\n  -F 'crop=true' \\\n  -F 'normalize=true' \\\n  -F 'normalization=true'`;

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible?.target.id) setActive(visible.target.id);
      },
      { rootMargin: "-20% 0px -65%", threshold: [0.1, 0.5] },
    );
    docSections.forEach(([id]) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });
    return () => observer.disconnect();
  }, []);

  const jump = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      const top = window.scrollY + element.getBoundingClientRect().top - 84;
      window.scrollTo({ top, left: 0, behavior: "smooth" });
    }
    setActive(id);
  };

  return (
    <div className="page docs-layout">
      <aside className="docs-toc">
        <span className="nav-kicker">На этой странице</span>
        {docSections.map(([id, label]) => (
          <button className={active === id ? "active" : ""} key={id} onClick={() => jump(id)} type="button">{label}</button>
        ))}
        <div className="docs-toc__card">
          <Icon name="play" size={18} />
          <strong>Запрос из браузера</strong>
          <p>В разделе тестирования можно загрузить изображение, задать параметры и посмотреть ответ.</p>
          <button onClick={() => navigate("playground", "mrz")} type="button">Открыть тестирование</button>
        </div>
      </aside>

      <article className="docs-content">
        <header className="docs-hero">
          <span className="eyebrow">OCR API · v1</span>
          <h1>Документация OCR API</h1>
          <p>Эндпоинты, параметры запросов, работа с задачами и правила обработки MRZ.</p>
          <div className="docs-badges"><span><Icon name="globe" size={15} /> HTTP / multipart</span><span><Icon name="shield" size={15} /> Без авторизации</span><span><Icon name="code" size={15} /> JSON + изображения</span></div>
        </header>

        <section className="doc-section" id="quickstart">
          <div className="doc-section__index">01</div>
          <div className="doc-section__body">
            <span className="eyebrow">Быстрый старт</span><h2>Первый запрос</h2>
            <p>Отправляйте запросы в формате <code>multipart/form-data</code>. Изображение передаётся в поле <code>image</code>, остальные параметры зависят от эндпоинта.</p>
            <div className="code-block">
              <div className="code-block__head"><span>Shell</span><button onClick={() => { void navigator.clipboard.writeText(curl); setCopied(true); window.setTimeout(() => setCopied(false), 1400); }} type="button"><Icon name={copied ? "check" : "copy"} size={15} />{copied ? "Скопировано" : "Копировать"}</button></div>
              <pre><code>{curl}</code></pre>
            </div>
            <div className="callout callout--info"><Icon name="info" /><div><strong>Адрес API</strong><p>Запросы идут на сервер, с которого открыт UI. При разработке через Vite запросы перенаправляются на <code>VITE_API_PROXY_TARGET</code>.</p></div></div>
          </div>
        </section>

        <section className="doc-section" id="endpoints">
          <div className="doc-section__index">02</div>
          <div className="doc-section__body">
            <span className="eyebrow">Эндпоинты</span><h2>Ответ в том же запросе</h2>
            <div className="endpoint-list">
              {tools.map((tool) => (
                <button key={tool.id} onClick={() => navigate("playground", tool.id)} type="button">
                  <span className="method">POST</span><code>{tool.endpoint}</code><span>{tool.description}</span><Icon name="arrow" size={17} />
                </button>
              ))}
            </div>
            <div className="callout callout--info"><Icon name="info" /><div><strong>Единый ответ MRZ</strong><p><code>/v1/ocr/mrz</code> распознаёт паспорт и ID-карту по строкам MRZ. Поля <code>document_type</code>, <code>document_series</code> и <code>document_number</code> имеют одинаковые имена для всех форматов. Значения <code>document_type</code>: <code>internal_passport</code>, <code>international_passport</code>, <code>identity_card</code>. У ID-карты <code>document_series</code> может быть <code>null</code>; в <code>mrz</code> возвращаются две или три строки.</p></div></div>
            <div className="callout callout--info"><Icon name="info" /><div><strong>Внутренний паспорт РФ</strong><p><code>/v1/ocr/passport/internal</code> принимает только фото в поле <code>image</code>. Russian OCR выбирается автоматически. В ответе — ФИО, пол, дата рождения, серия и номер паспорта, дата выдачи, код подразделения и <code>Issue_organization_ru</code> («Кем выдан»). Поля MRZ и журнал нормализации здесь отсутствуют.</p></div></div>
            <div className="parameter-grid">
              <div><strong><code>crop</code></strong><p>Для MRZ: при true выравнивает документ и вырезает MRZ перед OCR. При false движок получает исходное изображение.</p></div>
              <div><strong><code>normalize</code></strong><p>При true исправляет символы и дополняет MRZ по правилам страны. При false только удаляет пробелы, приводит текст к верхнему регистру и выбирает строки.</p></div>
              <div><strong><code>normalization</code></strong><p>При true возвращает журнал исправлений, при false — null в data.normalization. На исправление MRZ не влияет.</p></div>
              <div><strong><code>model</code></strong><p>Выбирает модель для «Всего текста». Для запросов к Qwen она должна быть включена и запущена на сервере. Внутренний паспорт РФ всегда использует Russian OCR и не принимает параметр model. Единый MRZ-эндпоинт использует MRZ Scanner и не принимает параметр model.</p></div>
              <div><strong><code>crop_document</code></strong><p>Для «Полей паспорта»: при true выравнивает и обрезает документ перед поиском полей. По умолчанию true.</p></div>
              <div><strong><code>padding</code></strong><p>Для «Полей паспорта»: отступ вокруг каждой вырезки в пикселях, от 0 до 64. По умолчанию 4.</p></div>
              <div><strong><code>ocr_model</code></strong><p>Для «Полей паспорта»: none — только вырезки; paddle — Paddle eslav v5, paddle-cyrillic — Paddle Cyrillic v5, russian-ocr-accurate — Russian OCR accurate. Эти три модели работают на CPU. qwen3-vl-4b и qwen3-vl-8b используют режим сервера. API по умолчанию не читает текст; в UI выбран Paddle eslav v5. Каждая вырезка распознаётся отдельно, без нормализации. Фото и подпись пропускаются.</p></div>
            </div>
          </div>
        </section>

        <section className="doc-section" id="tasks">
          <div className="doc-section__index">03</div>
          <div className="doc-section__body">
            <span className="eyebrow">Задачи</span><h2>Обработка через очередь</h2>
            <p>Эндпоинт создания задачи принимает те же multipart-поля, что и синхронная версия, и возвращает HTTP <code>202</code>. Используйте задачи для долгих операций: клиенту не нужно держать соединение открытым до конца OCR.</p>
            <div className="data-table-wrap"><table className="data-table">
              <thead><tr><th>Синхронно</th><th>Создать задачу</th></tr></thead>
              <tbody>{taskOperations.map(([sync, task]) => <tr key={task}><td><code>{sync}</code></td><td><code>{task}</code></td></tr>)}</tbody>
            </table></div>
            <div className="vertical-pipeline vertical-pipeline--compact">
              <div><span>01</span><div><h3>Создать задачу</h3><p><code>POST /v1/tasks/…</code> возвращает <code>&#123;"task_id":"…","status":"queued"&#125;</code> в поле data ответа.</p></div></div>
              <div><span>02</span><div><h3>Проверить статус</h3><p>Запрашивайте <code>GET /v1/tasks/&#123;task_id&#125;</code>. Возможные статусы: <code>queued</code>, <code>processing</code>, <code>done</code>, <code>failed</code>, <code>timeout</code>, <code>cancelled</code>.</p></div></div>
              <div><span>03</span><div><h3>Получить результат</h3><p><code>GET /v1/tasks/&#123;task_id&#125;/result</code> возвращает тот же JSON или JPEG, что и синхронный эндпоинт. Для отмены задачи: <code>POST /v1/tasks/&#123;task_id&#125;/cancel</code>.</p></div></div>
            </div>
            <div className="callout callout--info"><Icon name="info" /><div><strong>Лучший кадр</strong><p><code>/v1/ocr/best-frame</code> работает только синхронно. Создать задачу для этой операции нельзя.</p></div></div>
          </div>
        </section>

        <section className="doc-section" id="pipeline">
          <div className="doc-section__index">04</div>
          <div className="doc-section__body">
            <span className="eyebrow">Обработка MRZ</span><h2>От изображения к полям документа</h2>
            <div className="vertical-pipeline">
              {[
                ["01", "Проверка файла", "Сервис проверяет расширение, чтение изображения, количество страниц, размеры и число пикселей."],
                ["02", "Подготовка", "При crop=true находит границы документа, исправляет перспективу и ориентацию, затем вырезает MRZ."],
                ["03", "OCR", "MRZ Scanner распознаёт строки паспорта или ID-карты. Из строк удаляются пробелы, буквы переводятся в верхний регистр, затем выбираются две или три строки по формату MRZ."],
                ["04", "Определение формата", "По заголовку определяется страна и формат документа. Для неизвестного формата парсер не выбирается."],
                ["05", "Нормализация", "При normalize=true символы исправляются по правилам страны с учётом позиции в строке. Каждая замена и вставка записывается в журнал."],
                ["06", "Разбор полей", "Парсер выбранного формата извлекает поля документа и проверяет доступные контрольные цифры."],
              ].map(([number, title, text]) => (
                <div key={number}><span>{number}</span><div><h3>{title}</h3><p>{text}</p></div></div>
              ))}
            </div>
          </div>
        </section>

        <section className="doc-section" id="normalization">
          <div className="doc-section__index">05</div>
          <div className="doc-section__body">
            <span className="eyebrow">Нормализация MRZ</span><h2>Правила по странам</h2>
            <p>В MRZ буквы и цифры занимают фиксированные позиции. Правила ниже исправляют символы с учётом этих позиций. Для каждого исправления журнал содержит поле, позицию, исходный символ, замену и причину.</p>
            <div className="rule-grid">
              {countryRules.map((country, index) => (
                <details className="country-rule" key={`${country.name}-${index}`} open={index === 0}>
                  <summary><span className="country-code">{country.code}</span><span><strong>{country.name}</strong><code>{country.header}</code></span><Icon name="chevron" /></summary>
                  <ul>{country.rules.map((rule) => <li key={rule}><Icon name="check" size={15} />{rule}</li>)}</ul>
                </details>
              ))}
            </div>
            <div className="translation-map">
              <div><span>Цифра → буква</span><code>0→O · 1→I · 2→Z · 3→E · 4→A · 5→S · 6/9→G · 7→T · 8→B</code></div>
              <div><span>Буква → цифра</span><code>O→0 · I→1 · Z→2 · E→3 · A→4 · S→5 · G→6 · T→7 · B→8</code></div>
              <div><span>Пол</span><code>N/H→M · E→F</code></div>
            </div>
          </div>
        </section>

        <section className="doc-section" id="documents">
          <div className="doc-section__index">06</div>
          <div className="doc-section__body">
            <span className="eyebrow">Документы</span><h2>Поддерживаемые документы и форматы</h2>
            <div className="data-table-wrap"><table className="data-table">
              <thead><tr><th>Документ</th><th>Формат</th><th>Строки</th><th>Заголовок</th></tr></thead>
              <tbody>
                <tr><td>Международный паспорт</td><td>TD3</td><td>2 × 44</td><td><code>P&lt;ARM / P&lt;KAZ / P&lt;RUS / P&lt;TJK / PPTJK / P&lt;UZB</code></td></tr>
                <tr><td>Внутренний паспорт РФ</td><td>TD3-подобный</td><td>2 × 44</td><td><code>PNRUS</code></td></tr>
                <tr><td>ID-карта Казахстана</td><td>TD1</td><td>3 × 30</td><td><code>IDKAZ</code></td></tr>
                <tr><td>ID-карта Казахстана</td><td>TD2</td><td>2 × 36</td><td><code>IDKAZ</code></td></tr>
                <tr><td>ID-карта Кыргызстана</td><td>TD1</td><td>3 × 30</td><td><code>I&lt;KGZ / IDKGZ</code></td></tr>
                <tr><td>3-я страница паспорта РФ</td><td>Визуальная</td><td>—</td><td>Без MRZ</td></tr>
              </tbody>
            </table></div>
          </div>
        </section>

        <section className="doc-section" id="limits">
          <div className="doc-section__index">07</div>
          <div className="doc-section__body">
            <span className="eyebrow">Входные данные</span><h2>Форматы и лимиты</h2>
            <div className="limits-grid">
              <div><strong>75 <span>MiB</span></strong><p>Максимальный размер файла по умолчанию</p></div>
              <div><strong>12 000 <span>px</span></strong><p>Максимальная ширина и высота по умолчанию</p></div>
              <div><strong>50 <span>MP</span></strong><p>Максимальное число пикселей по умолчанию</p></div>
              <div><strong>1–3</strong><p>Допустимое число кадров для best-frame по умолчанию</p></div>
            </div>
            <p className="format-line"><strong>Поддерживаются:</strong> JPG, JPEG, PNG, BMP, WebP, TIFF. PDF, HEIC/HEIF и AVIF не принимаются. Один файл должен содержать одно изображение; передавайте страницы документа отдельными запросами.</p>
            <div className="callout callout--info"><Icon name="camera" /><div><strong>Съёмка с камеры</strong><p>В «Лучшем кадре» одно нажатие снимает три фото с коротким интервалом и отправляет их на сравнение. Рядом с результатом показаны отправленные снимки из браузера, выбранный кадр выделен. Сервер возвращает только одно изображение.</p><p>Предпросмотр с камеры и съёмка серии доступны на <code>localhost</code> и через HTTPS. На HTTP в локальной сети можно снять фото по отдельности через системную камеру или загрузить готовые файлы.</p></div></div>
          </div>
        </section>

        <section className="doc-section" id="errors">
          <div className="doc-section__index">08</div>
          <div className="doc-section__body">
            <span className="eyebrow">Ошибки</span><h2>Формат ответа с ошибкой</h2>
            <div className="error-example"><code>&#123; "success": false, "data": null, "error": &#123; "code": "OCR_FAILED", "message": "…" &#125; &#125;</code></div>
            <div className="data-table-wrap"><table className="data-table data-table--errors">
              <thead><tr><th>HTTP</th><th>Код</th><th>Причина</th></tr></thead>
              <tbody>{errors.map(([http, code, message]) => <tr key={code}><td><span className={`http-status http-status--${http[0]}`}>{http}</span></td><td><code>{code}</code></td><td>{message}</td></tr>)}</tbody>
            </table></div>
            <div className="callout callout--warning"><Icon name="clock" /><div><strong>Таймаут клиента</strong><p>Задайте таймаут больше серверного таймаута выбранной модели. Первый запрос может занять больше времени из-за загрузки модели. Для долгих операций используйте задачи.</p></div></div>
          </div>
        </section>
      </article>
    </div>
  );
}

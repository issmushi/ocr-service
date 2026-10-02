import { Icon, type IconName } from "../components/Icon";
import { tools } from "../config";
import type { HealthState, Section, ToolId } from "../types";

interface Props {
  health: HealthState;
  navigate: (section: Section, tool?: ToolId) => void;
}

const stats = [
  { value: String(tools.length), label: "эндпоинтов в UI", detail: "OCR, обрезка, выбор кадра" },
  { value: "5", label: "стран", detail: "паспортная MRZ" },
  { value: "2", label: "формата ID", detail: "TD1 и TD2" },
  { value: "75", suffix: "MiB", label: "на изображение", detail: "лимит по умолчанию" },
];

const stages: { icon: IconName; title: string; text: string }[] = [
  { icon: "image", title: "Файл", text: "Проверка формата файла, его размера и числа пикселей." },
  { icon: "crop", title: "Обрезка", text: "При crop=true: поиск документа, выравнивание и обрезка MRZ." },
  { icon: "scan", title: "Распознавание", text: "MRZ Scanner читает строки MRZ паспорта или ID-карты." },
  { icon: "spark", title: "Нормализация", text: "При normalize=true: замены похожих символов и дополнение MRZ по правилам страны." },
  { icon: "code", title: "Ответ", text: "MRZ, поля документа, список исправлений и meta с моделью и временем обработки." },
];

export function OverviewPage({ health, navigate }: Props) {
  return (
    <div className="page page--overview">
      <section className="hero">
        <div className="hero__copy">
          <span className="eyebrow"><span className="eyebrow__dot" />OCR API · v1</span>
          <h1>Документация<br />и <em>проверка OCR</em></h1>
          <p>Здесь собраны эндпоинты, параметры и правила обработки. В разделе тестирования можно отправить фото, сравнить модели и посмотреть ответ API.</p>
          <div className="hero__actions">
            <button className="button button--primary" onClick={() => navigate("playground", "passport-mrz")} type="button">
              <Icon name="play" size={18} />Открыть тестирование
            </button>
            <button className="button button--secondary" onClick={() => navigate("docs")} type="button">
              Документация<Icon name="arrow" size={18} />
            </button>
          </div>
        </div>
        <div className="hero-console" aria-label="Пример запроса и отдельных полей ответа OCR API">
          <div className="hero-console__head">
            <span><i /><i /><i /></span>
            <code>POST /v1/ocr/passport/mrz</code>
            <span className={`live-pill live-pill--${health.status}`}>
              {health.status === "online" ? "live" : health.status === "degraded" ? "degraded" : health.status === "checking" ? "checking" : "offline"}
            </span>
          </div>
          <div className="hero-console__body">
            <div className="code-line"><span className="code-num">01</span><span className="code-key">image</span><span className="code-muted">passport.jpg</span></div>
            <div className="code-line"><span className="code-num">02</span><span className="code-key">crop</span><span className="code-bool">true</span></div>
            <div className="code-line"><span className="code-num">03</span><span className="code-key">normalize</span><span className="code-bool">true</span></div>
            <div className="console-divider" />
            <div className="code-line"><span className="code-num">04</span><span className="code-punctuation">&#123;</span></div>
            <div className="code-line code-indent"><span className="code-num">05</span><span className="code-key">"success"</span><span className="code-bool">true</span><span className="code-punctuation">,</span></div>
            <div className="code-line code-indent"><span className="code-num">06</span><span className="code-key">"mrz_valid"</span><span className="code-bool">true</span><span className="code-punctuation">,</span></div>
            <div className="code-line code-indent"><span className="code-num">07</span><span className="code-key">"duration_seconds"</span><span className="code-number">1.57</span></div>
            <div className="code-line"><span className="code-num">08</span><span className="code-punctuation">&#125;</span></div>
          </div>
          <div className="hero-console__foot">
            <Icon name="check" size={14} /><span>Пример полей ответа</span><span>{health.latencyMs ? `/health: ${health.latencyMs} мс` : "GET /health"}</span>
          </div>
        </div>
      </section>

      <section className="stats-grid" aria-label="Документы и лимиты">
        {stats.map((stat) => (
          <div className="stat" key={stat.label}>
            <div><strong>{stat.value}</strong>{stat.suffix && <span>{stat.suffix}</span>}</div>
            <p>{stat.label}</p><small>{stat.detail}</small>
          </div>
        ))}
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div><span className="eyebrow">Эндпоинты</span><h2>Что можно проверить</h2></div>
          <button className="text-link" onClick={() => navigate("docs")} type="button">Параметры и примеры <Icon name="arrow" size={16} /></button>
        </div>
        <div className="tool-card-grid">
          {tools.map((tool) => (
            <button className="tool-card" key={tool.id} onClick={() => navigate("playground", tool.id)} type="button">
              <span className="tool-card__icon"><Icon name={tool.id.includes("crop") ? "crop" : tool.id === "best-frame" ? "frame" : tool.id === "passport-internal" ? "passport" : tool.id === "full-ocr" ? "file" : "scan"} /></span>
              <span className="method">POST</span>
              <h3>{tool.title}</h3>
              <code>{tool.endpoint}</code>
              <p>{tool.description}</p>
              <span className="tool-card__action">Проверить эндпоинт <Icon name="arrow" size={16} /></span>
            </button>
          ))}
        </div>
      </section>

      <section className="section-block pipeline-section">
        <div className="section-heading">
          <div><span className="eyebrow">Обработка</span><h2>Как разбираем MRZ</h2></div>
          <p>Обрезка и нормализация включаются флагами запроса.</p>
        </div>
        <div className="pipeline-flow">
          {stages.map((stage, index) => (
            <div className="pipeline-stage" key={stage.title}>
              <span className="pipeline-stage__number">0{index + 1}</span>
              <span className="pipeline-stage__icon"><Icon name={stage.icon} /></span>
              <h3>{stage.title}</h3><p>{stage.text}</p>
              {index < stages.length - 1 && <span className="pipeline-stage__line" />}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

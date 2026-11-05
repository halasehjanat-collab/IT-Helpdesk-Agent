import type { ChatCopy } from "./chatCopy"
import type { Diagnostics } from "./useHelpdeskChat"

export function formatValue(value: unknown): string {
  if (typeof value === "string") return value
  try {
    return JSON.stringify(value, null, 2) ?? String(value)
  } catch {
    return String(value)
  }
}

export function DataBlock({ value }: { value: unknown }) {
  return (
    <pre className="diag-data" dir="ltr">
      {formatValue(value)}
    </pre>
  )
}

export default function DiagnosticsPanel({
  diagnostics,
  t,
}: {
  diagnostics: Diagnostics
  t: ChatCopy
}) {
  const { citations, toolCalls, result, state } = diagnostics
  return (
    <div className="side-card diagnostics-card">
      <div className="card-title">
        <div>
          <h3>{t.diagnostics}</h3>
        </div>
      </div>
      <p className="diag-hint">{t.diagnosticsHint}</p>

      <details className="diag-section">
        <summary>
          {t.sources} <span>{citations.length}</span>
        </summary>
        {citations.length === 0 ? (
          <p className="diag-empty">{t.noSources}</p>
        ) : (
          <ul className="diag-list">
            {citations.map((citation, index) => (
              <li key={`${citation.id ?? ""}-${index}`}>
                {citation.id && <strong dir="ltr">{citation.id}</strong>}
                {citation.title && <span>{citation.title}</span>}
              </li>
            ))}
          </ul>
        )}
      </details>

      <details className="diag-section">
        <summary>
          {t.toolCalls} <span>{toolCalls.length}</span>
        </summary>
        {toolCalls.length === 0 ? (
          <p className="diag-empty">{t.noToolCalls}</p>
        ) : (
          <ul className="diag-list">
            {toolCalls.map((call, index) => {
              const status =
                call.success === undefined
                  ? "unknown"
                  : call.success
                    ? "ok"
                    : "fail"
              return (
                <li key={index} className={`tool-call ${status}`}>
                  <strong dir="ltr">{call.name || t.unnamedTool}</strong>
                  <span className="tool-status">
                    {status === "ok"
                      ? t.success
                      : status === "fail"
                        ? t.failed
                        : t.unknownStatus}
                  </span>
                  {call.durationMs !== undefined && (
                    <span className="tool-time" dir="ltr">
                      {Math.round(call.durationMs)} {t.ms}
                    </span>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </details>

      <details className="diag-section">
        <summary>{t.resultPanel}</summary>
        {result === null ? (
          <p className="diag-empty">{t.noResult}</p>
        ) : (
          <DataBlock value={result} />
        )}
      </details>

      <details className="diag-section">
        <summary>{t.statePanel}</summary>
        {state === null ? (
          <p className="diag-empty">{t.noState}</p>
        ) : (
          <DataBlock value={state} />
        )}
      </details>
    </div>
  )
}

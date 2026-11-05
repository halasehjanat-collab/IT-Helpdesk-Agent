import { normalizeTicketList, type TicketList } from "./tickets"

export const API_BASE_URL = ""

const REQUEST_TIMEOUT_MS = 60_000

export type ApiErrorKind = "network" | "timeout" | "http" | "invalid"

export class ApiError extends Error {
  kind: ApiErrorKind
  status?: number

  constructor(kind: ApiErrorKind, status?: number) {
    super(`API ${kind}${status ? ` ${status}` : ""}`)
    this.kind = kind
    this.status = status
  }
}

export type ActionId = string | number

export interface ChatRequest {
  session_id: string
  message: string
  user?: { email: string }
}

export interface ApproveRequest {
  session_id: string
  action_id: ActionId
  approved: boolean
}

export interface PendingAction {
  action_id: ActionId
  [key: string]: unknown
}

export interface Citation {
  id?: string
  title?: string
}

export interface ToolCall {
  name?: string
  success?: boolean
  durationMs?: number
}

export interface HelpdeskResponse {
  reply: string | null
  pendingAction: PendingAction | null
  pendingActionInvalid: boolean
  citations: Citation[]
  toolCalls: ToolCall[]
  result: unknown
  state: unknown
}

export interface TicketDraft {
  title: string
  summary: string
  subject: string
  category: string
  priority: string
  department: string
  requesterEmail: string
  description: string
  attemptedSteps: string[]
  kbReferences: string[]
}

type UnknownRecord = Record<string, unknown>

const isRecord = (value: unknown): value is UnknownRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isBlank = (value: unknown) =>
  value === null ||
  value === undefined ||
  value === "" ||
  (Array.isArray(value) && value.length === 0) ||
  (isRecord(value) && Object.keys(value).length === 0)

const asText = (value: unknown): string => {
  if (typeof value === "string") return value.trim()
  if (typeof value === "number" || typeof value === "boolean")
    return String(value)
  return ""
}

async function postJson(path: string, body: unknown): Promise<unknown> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw new ApiError("timeout")
    throw new ApiError("network")
  } finally {
    clearTimeout(timer)
  }

  if (!response.ok) throw new ApiError("http", response.status)

  try {
    return await response.json()
  } catch {
    throw new ApiError("invalid")
  }
}

function normalizeCitations(value: unknown): Citation[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item): Citation | null => {
      if (typeof item === "string") return { id: item }
      if (!isRecord(item)) return null
      const id = asText(item.id ?? item.doc_id ?? item.source_id)
      const title = asText(item.title ?? item.name)
      return id || title ? { id: id || undefined, title: title || undefined } : null
    })
    .filter((item): item is Citation => item !== null)
}

function normalizeToolCalls(value: unknown): ToolCall[] {
  if (!Array.isArray(value)) return []
  return value
    .filter(isRecord)
    .map((item) => {
      const name = asText(item.name ?? item.tool ?? item.tool_name)
      const rawStatus = item.success ?? item.ok ?? item.status
      let success: boolean | undefined
      if (typeof rawStatus === "boolean") success = rawStatus
      else if (typeof rawStatus === "string") {
        const status = rawStatus.toLowerCase()
        if (["success", "ok", "succeeded", "completed"].includes(status))
          success = true
        else if (["error", "failed", "failure"].includes(status))
          success = false
      }
      if (success === undefined && item.error) success = false
      const rawDuration =
        item.duration_ms ?? item.elapsed_ms ?? item.latency_ms ?? item.time_ms
      const durationMs =
        typeof rawDuration === "number" && Number.isFinite(rawDuration)
          ? rawDuration
          : undefined
      return { name: name || undefined, success, durationMs }
    })
}

function normalizeResponse(body: unknown): HelpdeskResponse {
  if (!isRecord(body)) throw new ApiError("invalid")

  const replySource = body.reply ?? body.response ?? body.answer ?? body.message
  const reply = typeof replySource === "string" ? replySource : null

  let pendingAction: PendingAction | null = null
  let pendingActionInvalid = false
  const rawAction = body.pending_action
  if (isRecord(rawAction)) {
    const id = rawAction.action_id
    if (
      (typeof id === "string" && id.trim()) ||
      (typeof id === "number" && Number.isFinite(id))
    )
      pendingAction = { ...rawAction, action_id: id }
    else pendingActionInvalid = true
  } else if (!isBlank(rawAction)) pendingActionInvalid = true

  return {
    reply,
    pendingAction,
    pendingActionInvalid,
    citations: normalizeCitations(body.citations),
    toolCalls: normalizeToolCalls(body.tool_calls),
    result: isBlank(body.result) ? null : body.result,
    state: isBlank(body.state) ? null : body.state,
  }
}

export async function postChat(request: ChatRequest): Promise<HelpdeskResponse> {
  return normalizeResponse(await postJson("/api/chat", request))
}

export async function postApprove(
  request: ApproveRequest,
): Promise<HelpdeskResponse> {
  return normalizeResponse(await postJson("/api/approve", request))
}

// Read-only list of the requester's tickets from the backend (never creates).
export async function fetchTickets(email: string): Promise<TicketList> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  let response: Response
  try {
    response = await fetch(
      `${API_BASE_URL}/api/tickets?requester_email=${encodeURIComponent(email)}`,
      { signal: controller.signal },
    )
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw new ApiError("timeout")
    throw new ApiError("network")
  } finally {
    clearTimeout(timer)
  }
  if (!response.ok) throw new ApiError("http", response.status)
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new ApiError("invalid")
  }
  const list = normalizeTicketList(body)
  if (!list) throw new ApiError("invalid")
  return list
}

export interface Employee {
  email: string
  name: string
  department: string
  title: string
}

// Read-only directory check: the name shown in the UI always comes from the
// backend, never from what the user typed.
export async function identifyEmployee(email: string): Promise<Employee | null> {
  const body = await postJson("/api/identify", { email: email.trim().toLowerCase() })
  if (!isRecord(body) || body.verified !== true) return null
  const name = asText(body.name)
  const verifiedEmail = asText(body.email)
  if (!name || !verifiedEmail) return null
  return {
    email: verifiedEmail.toLowerCase(),
    name,
    department: asText(body.department),
    title: asText(body.title),
  }
}

const DRAFT_CONTAINERS = [
  "payload",
  "args",
  "arguments",
  "draft",
  "ticket",
  "data",
  "params",
]

function pick(sources: UnknownRecord[], keys: string[]): unknown {
  for (const source of sources)
    for (const key of keys)
      if (!isBlank(source[key])) return source[key]
  return undefined
}

function toTextList(value: unknown): string[] {
  if (isBlank(value)) return []
  const items = Array.isArray(value) ? value : [value]
  return items
    .map((item) => {
      if (isRecord(item)) {
        const id = asText(item.id ?? item.doc_id)
        const title = asText(item.title ?? item.name)
        return [id, title].filter(Boolean).join(" · ")
      }
      return asText(item)
    })
    .filter(Boolean)
}

export function extractDraft(action: PendingAction): TicketDraft {
  const sources: UnknownRecord[] = [
    action,
    ...DRAFT_CONTAINERS.map((key) => action[key]).filter(isRecord),
  ]
  const text = (...keys: string[]) => asText(pick(sources, keys))
  const summary = text("summary")
  const subject = text("subject", "title")
  return {
    title: summary || subject,
    summary,
    subject,
    category: text("category"),
    priority: text("priority"),
    department: text("department"),
    requesterEmail: text("requester_email", "requester", "email"),
    description: text("description"),
    attemptedSteps: toTextList(
      pick(sources, ["attempted_steps", "steps_attempted", "troubleshooting"]),
    ),
    kbReferences: toTextList(
      pick(sources, [
        "kb_references",
        "kb_refs",
        "kb_articles",
        "knowledge_base_references",
        "citations",
      ]),
    ),
  }
}

// Detects an explicit failure reported by the backend inside a 2xx response.
export function resultIndicatesFailure(result: unknown): boolean {
  if (!isRecord(result)) return false
  if (result.success === false || result.ok === false) return true
  if (!isBlank(result.error)) return true
  const status = asText(result.status).toLowerCase()
  return ["error", "failed", "failure"].includes(status)
}

// The backend confirms creation with state "ticket_created" (or "already_approved"
// on an idempotent repeat) plus result.ticket_id. HTTP 200 alone proves nothing.
export function approvalOutcomeType(
  response: HelpdeskResponse,
  approved: boolean,
): "approved" | "rejected" | "failed" {
  const state = typeof response.state === "string" ? response.state : ""
  if (!approved)
    return state === "rejected" || state === "already_rejected"
      ? "rejected"
      : "failed"
  const hasTicket = findTicketId(response.result) !== ""
  const created =
    state === "ticket_created" ||
    (state === "already_approved" && hasTicket)
  return created && hasTicket && !resultIndicatesFailure(response.result)
    ? "approved"
    : "failed"
}

export function findTicketId(result: unknown): string {
  if (!isRecord(result)) return ""
  const nested = isRecord(result.ticket) ? result.ticket : undefined
  return asText(
    result.ticket_id ??
      result.ticket_number ??
      nested?.id ??
      nested?.key ??
      nested?.number,
  )
}

export const isEmailValid = (email: string) =>
  /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim())

export function createSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto)
    return crypto.randomUUID()
  return `session-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

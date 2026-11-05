// Pure helpers for the live ticket list. No imports, so they run under `node --test`.

export interface TicketRow {
  id: string
  title: string
  category: string
  priority: string
  status: string
  date: string
}

export interface TicketList {
  tickets: TicketRow[]
  total: number
}

export interface TicketStats {
  total: number
  open: number
  resolved: number
  closed: number
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const text = (value: unknown) => (typeof value === "string" ? value.trim() : "")

const STATUS_LABEL: Record<string, string> = {
  open: "Open",
  resolved: "Resolved",
  closed: "Closed",
}

// Reuses the existing High / Medium / Low priority styles.
const PRIORITY_LABEL: Record<string, string> = {
  p1: "High",
  p2: "High",
  p3: "Medium",
  p4: "Low",
}

const CATEGORY_LABEL: Record<string, string> = {
  network_access: "Network",
  identity_access: "Access",
  hardware: "Hardware",
  software: "Software",
  email_collab: "Email",
  security: "Security",
  onboarding: "Onboarding",
  other: "Other",
}

export function formatTicketDate(value: string): string {
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
}

// Returns null for anything that is not a well-formed list response.
export function normalizeTicketList(body: unknown): TicketList | null {
  if (!isRecord(body) || body.state !== "ok" || !Array.isArray(body.tickets))
    return null
  const seen = new Set<string>()
  const tickets: TicketRow[] = []
  for (const raw of body.tickets) {
    if (!isRecord(raw)) continue
    const id = text(raw.ticket_id)
    if (!id || seen.has(id)) continue
    seen.add(id)
    const status = text(raw.status).toLowerCase()
    const category = text(raw.category)
    tickets.push({
      id,
      title: text(raw.subject) || id,
      category: CATEGORY_LABEL[category] ?? (category || "Other"),
      priority: PRIORITY_LABEL[text(raw.priority).toLowerCase()] ?? "Medium",
      status: STATUS_LABEL[status] ?? (status || "Open"),
      date: formatTicketDate(text(raw.created_at)),
    })
  }
  const total =
    typeof body.total === "number" && body.total >= tickets.length
      ? body.total
      : tickets.length
  return { tickets, total }
}

export function computeStats(list: TicketList): TicketStats {
  const count = (status: string) =>
    list.tickets.filter((ticket) => ticket.status === status).length
  return {
    total: list.total,
    open: count("Open"),
    resolved: count("Resolved"),
    closed: count("Closed"),
  }
}

// The list is refreshed only when the backend confirmed a ticket was created.
export const shouldRefreshTickets = (outcomeType: string | undefined) =>
  outcomeType === "approved"

// The backend returns a draft only on the turn that creates it and keeps it until it is
// decided. A chat reply without a draft must therefore keep the one already shown, while
// an approve/reject response (keepUndecided = false) clears it.
export function nextPendingAction<T>(
  current: T | null,
  incoming: T | null,
  keepUndecided: boolean,
): T | null {
  return incoming ?? (keepUndecided ? current : null)
}

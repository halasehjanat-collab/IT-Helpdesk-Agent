import { useCallback, useEffect, useRef, useState } from "react"
import { fetchTickets, isEmailValid } from "./api"
import { shouldRefreshTickets, type TicketList } from "./tickets"
import type { DraftOutcome } from "./useHelpdeskChat"

export type TicketsStatus = "no-email" | "loading" | "ready" | "error"

// Loads the requester's real tickets and reloads them after a confirmed creation.
export function useTickets(email: string, outcome: DraftOutcome | null) {
  const [list, setList] = useState<TicketList | null>(null)
  const [status, setStatus] = useState<TicketsStatus>("no-email")
  const requestId = useRef(0)
  const normalized = email.trim().toLowerCase()

  const refresh = useCallback(async () => {
    const id = ++requestId.current
    if (!normalized || !isEmailValid(normalized)) {
      setList(null)
      setStatus("no-email")
      return
    }
    setStatus("loading")
    try {
      const next = await fetchTickets(normalized)
      if (id !== requestId.current) return
      setList(next)
      setStatus("ready")
    } catch {
      if (id !== requestId.current) return
      setStatus("error")
    }
  }, [normalized])

  useEffect(() => {
    // Debounced so typing an address does not query on every keystroke.
    const timer = setTimeout(() => void refresh(), 400)
    return () => clearTimeout(timer)
  }, [refresh])

  const confirmedOutcome = shouldRefreshTickets(outcome?.type) ? outcome : null
  useEffect(() => {
    if (confirmedOutcome) void refresh()
  }, [confirmedOutcome, refresh])

  return { list, status, refresh }
}

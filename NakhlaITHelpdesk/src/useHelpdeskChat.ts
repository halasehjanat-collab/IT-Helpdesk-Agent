import { useCallback, useRef, useState } from "react"
import {
  ApiError,
  createSessionId,
  isEmailValid,
  postApprove,
  postChat,
  approvalOutcomeType,
  type ApiErrorKind,
  type Citation,
  type HelpdeskResponse,
  type PendingAction,
  type ToolCall,
} from "./api"
import { nextPendingAction } from "./tickets"

export interface ChatMessage {
  id: number
  role: "user" | "assistant"
  text: string
  time: Date
}

export interface ChatError {
  scope: "chat" | "approval"
  kind: ApiErrorKind | "email"
  status?: number
}

export interface Diagnostics {
  citations: Citation[]
  toolCalls: ToolCall[]
  result: unknown
  state: unknown
}

export interface DraftOutcome {
  type: "approved" | "rejected" | "failed"
  result: unknown
  reply: string | null
}

export type Busy = "chat" | "approve" | "reject" | null

const emptyDiagnostics: Diagnostics = {
  citations: [],
  toolCalls: [],
  result: null,
  state: null,
}

export function useHelpdeskChat() {
  const [sessionId, setSessionId] = useState(createSessionId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null)
  const [pendingActionInvalid, setPendingActionInvalid] = useState(false)
  const [diagnostics, setDiagnostics] = useState<Diagnostics>(emptyDiagnostics)
  const [busy, setBusy] = useState<Busy>(null)
  const [error, setError] = useState<ChatError | null>(null)
  const [outcome, setOutcome] = useState<DraftOutcome | null>(null)
  const [failedMessage, setFailedMessage] = useState<string | null>(null)
  const [email, setEmail] = useState("")
  const lock = useRef(false)
  const nextId = useRef(1)
  // Bumped on sign-out so a reply still in flight is never shown to the next user.
  const epoch = useRef(0)

  const addMessage = useCallback((role: ChatMessage["role"], text: string) => {
    setMessages((current) => [
      ...current,
      { id: nextId.current++, role, text, time: new Date() },
    ])
  }, [])

  // The backend returns a draft only on the turn that creates it and keeps it until it is
  // decided, so an ordinary chat reply must not clear a draft that is still pending.
  const applyResponse = useCallback(
    (response: HelpdeskResponse, keepUndecided = false) => {
    setPendingAction((current) =>
      nextPendingAction(current, response.pendingAction, keepUndecided),
    )
    setPendingActionInvalid(response.pendingActionInvalid)
    setDiagnostics({
      citations: response.citations,
      toolCalls: response.toolCalls,
      result: response.result,
      state: response.state,
    })
    },
    [],
  )

  const toError = (scope: ChatError["scope"], caught: unknown): ChatError =>
    caught instanceof ApiError
      ? { scope, kind: caught.kind, status: caught.status }
      : { scope, kind: "invalid" }

  const send = useCallback(
    async (rawText: string, rawEmail: string): Promise<boolean> => {
      const message = rawText.trim()
      const email = rawEmail.trim().toLowerCase()
      if (!message || lock.current) return false
      if (email && !isEmailValid(email)) {
        setError({ scope: "chat", kind: "email" })
        return false
      }

      lock.current = true
      const startedIn = epoch.current
      setBusy("chat")
      setError(null)
      setFailedMessage(null)
      setOutcome(null)
      addMessage("user", message)
      try {
        const response = await postChat({
          session_id: sessionId,
          message,
          ...(email ? { user: { email } } : {}),
        })
        if (epoch.current !== startedIn) return false
        if (response.reply) addMessage("assistant", response.reply)
        applyResponse(response, true)
        return true
      } catch (caught) {
        if (epoch.current !== startedIn) return false
        setError(toError("chat", caught))
        setFailedMessage(message)
        return false
      } finally {
        if (epoch.current === startedIn) {
          lock.current = false
          setBusy(null)
        }
      }
    },
    [sessionId, addMessage, applyResponse],
  )

  const decide = useCallback(
    async (approved: boolean) => {
      if (!pendingAction || lock.current) return
      lock.current = true
      const startedIn = epoch.current
      setBusy(approved ? "approve" : "reject")
      setError(null)
      setOutcome(null)
      try {
        const response = await postApprove({
          session_id: sessionId,
          action_id: pendingAction.action_id,
          approved,
        })
        if (epoch.current !== startedIn) return
        if (response.reply) addMessage("assistant", response.reply)
        applyResponse(response)
        const type = approvalOutcomeType(response, approved)
        // A failed creation can be retried with the same approved action.
        if (response.state === "ticket_creation_failed")
          setPendingAction(pendingAction)
        setOutcome({
          type,
          result: response.result,
          reply: response.reply,
        })
      } catch (caught) {
        if (epoch.current !== startedIn) return
        // The draft stays in place so the decision can be retried safely.
        setError(toError("approval", caught))
      } finally {
        if (epoch.current === startedIn) {
          lock.current = false
          setBusy(null)
        }
      }
    },
    [pendingAction, sessionId, addMessage, applyResponse],
  )

  const reset = useCallback(() => {
    if (lock.current) return
    setSessionId(createSessionId())
    setMessages([])
    setPendingAction(null)
    setPendingActionInvalid(false)
    setDiagnostics(emptyDiagnostics)
    setError(null)
    setOutcome(null)
    setFailedMessage(null)
  }, [])

  // Full wipe for sign-out / account switch: new session, no messages, no email.
  const signOut = useCallback(() => {
    epoch.current += 1
    lock.current = false
    setBusy(null)
    setSessionId(createSessionId())
    setMessages([])
    setPendingAction(null)
    setPendingActionInvalid(false)
    setDiagnostics(emptyDiagnostics)
    setError(null)
    setOutcome(null)
    setFailedMessage(null)
    setEmail("")
  }, [])

  return {
    signOut,
    sessionId,
    messages,
    pendingAction,
    pendingActionInvalid,
    diagnostics,
    busy,
    error,
    outcome,
    failedMessage,
    email,
    setEmail,
    send,
    approve: () => decide(true),
    reject: () => decide(false),
    reset,
    clearOutcome: () => setOutcome(null),
    clearError: () => setError(null),
  }
}

export type HelpdeskChat = ReturnType<typeof useHelpdeskChat>

// Run with: node --test tests/
import test from "node:test"
import assert from "node:assert/strict"
import {
  computeStats,
  nextPendingAction,
  normalizeTicketList,
  shouldRefreshTickets,
} from "../src/tickets.ts"

const row = (id, extra = {}) => ({
  ticket_id: id,
  subject: `Subject ${id}`,
  category: "network_access",
  priority: "P2",
  status: "open",
  created_at: "2025-01-01T10:00:00Z",
  ...extra,
})

test("normalizes a valid list and maps labels", () => {
  const list = normalizeTicketList({ state: "ok", tickets: [row("IT-1")], total: 1 })
  assert.equal(list.tickets.length, 1)
  assert.deepEqual(
    [list.tickets[0].category, list.tickets[0].priority, list.tickets[0].status],
    ["Network", "High", "Open"],
  )
})

test("rejects malformed responses", () => {
  assert.equal(normalizeTicketList(null), null)
  assert.equal(normalizeTicketList({ state: "tickets_unavailable" }), null)
  assert.equal(normalizeTicketList({ state: "ok", tickets: "x" }), null)
})

test("skips invalid rows and de-duplicates ids", () => {
  const list = normalizeTicketList({
    state: "ok",
    tickets: [row("IT-1"), row("IT-1"), { subject: "no id" }, 5],
  })
  assert.deepEqual(list.tickets.map((t) => t.id), ["IT-1"])
  assert.equal(list.total, 1)
})

test("stats are computed from real rows and total comes from the backend", () => {
  const list = normalizeTicketList({
    state: "ok",
    total: 7,
    tickets: [row("A"), row("B", { status: "resolved" }), row("C", { status: "closed" })],
  })
  assert.deepEqual(computeStats(list), { total: 7, open: 1, resolved: 1, closed: 1 })
})

test("count increases by exactly one after a created ticket", () => {
  const before = normalizeTicketList({ state: "ok", tickets: [row("A")], total: 1 })
  const after = normalizeTicketList({ state: "ok", tickets: [row("A"), row("B")], total: 2 })
  assert.equal(computeStats(after).total - computeStats(before).total, 1)
})

test("refresh happens only for a confirmed approval", () => {
  assert.equal(shouldRefreshTickets("approved"), true)
  for (const type of ["failed", "rejected", undefined])
    assert.equal(shouldRefreshTickets(type), false)
})

test("a chat reply without a draft keeps the pending draft", () => {
  const draft = { action_id: "a1" }
  assert.equal(nextPendingAction(draft, null, true), draft)
})

test("a new draft replaces the pending one", () => {
  const next = { action_id: "a2" }
  assert.equal(nextPendingAction({ action_id: "a1" }, next, true), next)
})

test("approve/reject responses clear the draft", () => {
  assert.equal(nextPendingAction({ action_id: "a1" }, null, false), null)
})

test("no draft stays no draft", () => {
  assert.equal(nextPendingAction(null, null, true), null)
})

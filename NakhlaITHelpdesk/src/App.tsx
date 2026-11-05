import { useEffect, useMemo, useState } from "react"
import { LocalizedContent } from "./i18n"
import { extractDraft, findTicketId, identifyEmployee, isEmailValid, type Employee } from "./api"
import { computeStats, type TicketRow } from "./tickets"
import { useTickets } from "./useTickets"
import { chatCopy } from "./chatCopy"
import DiagnosticsPanel, { DataBlock } from "./DiagnosticsPanel"
import { useHelpdeskChat, type HelpdeskChat } from "./useHelpdeskChat"
import type { Language } from "./types"

type Role = "employee" | "support" | "admin"
type View = "chat" | "tickets" | "ticket" | "support" | "admin" | "users" | "reports"

type IconName = "arrow" | "bell" | "book" | "bot" | "chart" | "check" | "chevron" | "clock" | "download" | "filter" | "globe" | "headset" | "home" | "laptop" | "lock" | "logout" | "menu" | "message" | "more" | "paperclip" | "report" | "search" | "send" | "shield" | "sparkles" | "ticket" | "trend" | "user" | "users" | "x"

const copy = {
  en: {
    welcome: "Welcome to Nakhla Helpdesk",
    loginSubtitle: "Choose your workspace to explore the support experience.",
    employee: "Employee",
    employeeDesc: "Get AI-powered help and manage your requests.",
    support: "IT Support",
    supportDesc: "Resolve assigned tickets and support your team.",
    admin: "Domain Admin",
    adminDesc: "Manage operations, users, and performance.",
    continue: "Continue to workspace",
    prototype: "Employee AI chat connects to the helpdesk backend · other dashboards show sample data",
    overview: "Overview",
    aiAssistant: "AI Assistant",
    myTickets: "My Tickets",
    assigned: "Assigned Tickets",
    users: "User Management",
    reports: "Reports",
    goodMorning: "Good morning, {name}",
    chatSubtitle: "How can I help with your IT today?",
    newConversation: "New conversation",
    askPlaceholder: "Describe your IT issue...",
    suggested: "Suggested issues",
    openTicket: "Open ticket",
    viewAll: "View all",
    signOut: "Back to role selection",
  },
  ar: {
    welcome: "مرحباً بك في مركز دعم نخلة",
    loginSubtitle: "اختر مساحة العمل لاستكشاف تجربة الدعم.",
    employee: "الموظف",
    employeeDesc: "احصل على مساعدة ذكية وأدِر طلباتك.",
    support: "الدعم التقني",
    supportDesc: "عالج التذاكر المعيّنة وادعم فريقك.",
    admin: "مدير النطاق",
    adminDesc: "أدِر العمليات والمستخدمين والأداء.",
    continue: "الدخول إلى مساحة العمل",
    prototype: "محادثة الموظف الذكية متصلة بخادم الدعم · لوحات المعلومات الأخرى تعرض بيانات تجريبية",
    overview: "نظرة عامة",
    aiAssistant: "المساعد الذكي",
    myTickets: "تذاكري",
    assigned: "التذاكر المعيّنة",
    users: "إدارة المستخدمين",
    reports: "التقارير",
    goodMorning: "صباح الخير، {name}",
    chatSubtitle: "كيف يمكنني مساعدتك تقنياً اليوم؟",
    newConversation: "محادثة جديدة",
    askPlaceholder: "صِف مشكلتك التقنية...",
    suggested: "مشكلات مقترحة",
    openTicket: "فتح تذكرة",
    viewAll: "عرض الكل",
    signOut: "العودة لاختيار الدور",
  },
  he: {
    welcome: "ברוכים הבאים לתמיכה של Nakhla",
    loginSubtitle: "בחרו סביבת עבודה כדי לחקור את חוויית התמיכה.",
    employee: "עובד",
    employeeDesc: "קבלו עזרה חכמה ונהלו את הבקשות שלכם.",
    support: "תמיכת IT",
    supportDesc: "טפלו בכרטיסים שהוקצו ותמכו בצוות.",
    admin: "מנהל דומיין",
    adminDesc: "נהלו תפעול, משתמשים וביצועים.",
    continue: "המשך לסביבת העבודה",
    prototype: "צ׳אט ה-AI לעובדים מחובר לשרת התמיכה · שאר הלוחות מציגים נתוני דוגמה",
    overview: "סקירה",
    aiAssistant: "עוזר חכם",
    myTickets: "הכרטיסים שלי",
    assigned: "כרטיסים שהוקצו",
    users: "ניהול משתמשים",
    reports: "דוחות",
    goodMorning: "בוקר טוב, {name}",
    chatSubtitle: "איך אוכל לעזור ב-IT היום?",
    newConversation: "שיחה חדשה",
    askPlaceholder: "תארו את בעיית ה-IT...",
    suggested: "בעיות מוצעות",
    openTicket: "פתיחת כרטיס",
    viewAll: "הצג הכל",
    signOut: "חזרה לבחירת תפקיד",
  },
}

const tickets = [
  {
    id: "IT-2841",
    title: "VPN connection drops frequently",
    category: "Network",
    priority: "High",
    status: "In progress",
    date: "Today, 9:42 AM",
  },
  {
    id: "IT-2837",
    title: "Microsoft Teams audio not working",
    category: "Software",
    priority: "Medium",
    status: "Waiting",
    date: "Yesterday, 3:15 PM",
  },
  {
    id: "IT-2819",
    title: "Request access to Finance shared drive",
    category: "Access",
    priority: "Low",
    status: "Resolved",
    date: "May 12, 11:30 AM",
  },
  {
    id: "IT-2802",
    title: "Laptop battery draining quickly",
    category: "Hardware",
    priority: "Medium",
    status: "Resolved",
    date: "May 10, 2:08 PM",
  },
]

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, React.ReactNode> = {
    arrow: (
      <>
        <path d="M5 12h14" />
        <path d="m13 6 6 6-6 6" />
      </>
    ),
    bell: (
      <>
        <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9" />
        <path d="M10 21h4" />
      </>
    ),
    book: (
      <>
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2Z" />
      </>
    ),
    bot: (
      <>
        <rect x="4" y="7" width="16" height="13" rx="3" />
        <path d="M12 3v4M8 12h.01M16 12h.01M8 16h8" />
      </>
    ),
    chart: (
      <>
        <path d="M3 3v18h18" />
        <path d="m7 16 4-5 4 3 5-7" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    chevron: <path d="m9 18 6-6-6-6" />,
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    download: (
      <>
        <path d="M12 3v12m0 0 5-5m-5 5-5-5" />
        <path d="M5 21h14" />
      </>
    ),
    filter: <path d="M4 5h16l-6 7v5l-4 2v-7Z" />,
    globe: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18" />
      </>
    ),
    headset: (
      <>
        <path d="M4 14v-2a8 8 0 0 1 16 0v2" />
        <path d="M4 14a2 2 0 0 1 2-2h1v6H6a2 2 0 0 1-2-2Zm16 0a2 2 0 0 0-2-2h-1v6h1a2 2 0 0 0 2-2Z" />
        <path d="M17 18c0 2-2 3-5 3" />
      </>
    ),
    home: (
      <>
        <path d="m3 11 9-8 9 8" />
        <path d="M5 10v10h14V10M9 20v-6h6v6" />
      </>
    ),
    laptop: (
      <>
        <rect x="4" y="4" width="16" height="12" rx="2" />
        <path d="M2 20h20" />
      </>
    ),
    lock: (
      <>
        <rect x="4" y="10" width="16" height="11" rx="2" />
        <path d="M8 10V7a4 4 0 0 1 8 0v3" />
      </>
    ),
    logout: (
      <>
        <path d="M10 17l5-5-5-5M15 12H3" />
        <path d="M14 3h5a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-5" />
      </>
    ),
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    message: (
      <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4Z" />
    ),
    more: (
      <>
        <circle cx="5" cy="12" r="1" />
        <circle cx="12" cy="12" r="1" />
        <circle cx="19" cy="12" r="1" />
      </>
    ),
    paperclip: (
      <path d="m20 11-8.5 8.5a5 5 0 0 1-7-7L14 3a3.5 3.5 0 0 1 5 5l-9.5 9.5a2 2 0 0 1-3-3L15 6" />
    ),
    report: (
      <>
        <path d="M6 2h9l4 4v16H6Z" />
        <path d="M14 2v5h5M9 13h6M9 17h6" />
      </>
    ),
    search: (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-4-4" />
      </>
    ),
    send: (
      <>
        <path d="m22 2-7 20-4-9-9-4Z" />
        <path d="M22 2 11 13" />
      </>
    ),
    shield: (
      <>
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" />
        <path d="m9 12 2 2 4-4" />
      </>
    ),
    sparkles: (
      <>
        <path d="m12 3-1.2 3.8L7 8l3.8 1.2L12 13l1.2-3.8L17 8l-3.8-1.2ZM5 15l-.7 2.3L2 18l2.3.7L5 21l.7-2.3L8 18l-2.3-.7ZM19 13l-.7 2.3L16 16l2.3.7L19 19l.7-2.3L22 16l-2.3-.7Z" />
      </>
    ),
    ticket: (
      <path d="M3 6a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v3a3 3 0 0 0 0 6v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-3a3 3 0 0 0 0-6Z" />
    ),
    trend: (
      <>
        <path d="m3 17 6-6 4 4 8-9" />
        <path d="M15 6h6v6" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21a8 8 0 0 1 16 0" />
      </>
    ),
    users: (
      <>
        <path d="M16 21a6 6 0 0 0-12 0M10 11a4 4 0 1 0 0-8" />
        <path d="M17 11a4 4 0 0 0 0-8M20 21a6 6 0 0 0-4-5.5" />
      </>
    ),
    x: <path d="m6 6 12 12M18 6 6 18" />,
  }
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  )
}

function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="logo">
      <div className="logo-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      {!compact && (
        <div>
          <strong>NAKHLA</strong>
          <small>IT HELPDESK</small>
        </div>
      )}
    </div>
  )
}

function PalmGrove({ className = "" }: { className?: string }) {
  return (
    <svg
      className={`palm-grove ${className}`}
      viewBox="0 0 720 430"
      fill="none"
      aria-hidden="true"
    >
      <path
        className="palm-ground"
        d="M-20 410C95 352 181 391 274 374C381 354 470 299 740 370V450H-20Z"
      />
      <g className="palm-back">
        <path d="M531 379C541 298 536 237 511 171L524 167C557 240 562 312 550 389Z" />
        <path d="M518 178C472 158 445 126 423 88C465 100 500 124 522 165C489 120 480 78 481 39C511 74 526 116 524 164C535 111 556 76 587 47C584 95 566 137 531 169C574 130 613 118 650 117C621 147 582 168 533 176C582 167 622 179 653 204C609 208 572 198 529 183C559 210 574 242 577 278C547 253 527 221 520 185C501 222 475 245 443 257C458 218 482 191 518 178Z" />
      </g>
      <g className="palm-front">
        <path d="M178 420C196 312 202 225 185 127L202 123C228 226 222 327 207 430Z" />
        <path d="M193 136C127 121 86 84 50 38C108 47 158 75 194 119C155 67 147 20 153-26C188 17 204 64 199 119C216 60 246 21 288-8C278 47 250 93 207 126C262 86 314 80 358 92C313 121 265 136 208 135C265 141 305 165 333 200C282 196 239 177 201 143C230 182 239 222 230 263C197 226 181 185 193 136Z" />
        <path d="M197 131C183 116 172 96 166 77C184 85 197 103 201 125C212 103 227 88 246 80C240 102 227 119 204 132C226 133 243 140 257 153C233 157 216 150 197 131Z" />
      </g>
      <g className="palm-small">
        <path d="M658 419C663 370 660 334 647 299L655 297C674 335 677 376 671 426Z" />
        <path d="M651 303C623 293 605 274 593 250C620 256 640 268 653 295C637 266 637 242 642 222C658 242 662 267 656 295C669 268 687 253 707 245C697 269 681 289 658 300C686 286 710 285 733 292C709 306 685 310 658 305C683 312 699 327 708 346C684 338 666 324 651 303Z" />
      </g>
    </svg>
  )
}

function BotanicalCorner({ className = "" }: { className?: string }) {
  return (
    <svg
      className={`botanical-corner ${className}`}
      viewBox="0 0 390 300"
      fill="none"
      aria-hidden="true"
    >
      <path
        className="leaf-soft"
        d="M382 8C282 22 208 85 196 193C277 184 352 130 382 8Z"
      />
      <path className="leaf-line" d="M370 23C315 75 261 126 204 185" />
      <path
        className="leaf-soft leaf-peach"
        d="M324 0C247 36 207 94 219 170C280 144 326 90 324 0Z"
      />
      <path
        className="leaf-line peach-line"
        d="M318 14C283 64 252 111 223 159"
      />
      <path
        className="leaf-deep"
        d="M390 125C325 115 273 137 242 194C296 212 357 187 390 125Z"
      />
      <path
        className="leaf-line deep-line"
        d="M377 132C330 151 291 171 251 192"
      />
      <path
        className="leaf-dot"
        d="M289 73C281 56 283 38 296 23C306 42 304 59 289 73Z"
      />
    </svg>
  )
}

function LanguageSwitcher({
  language,
  setLanguage,
  compact = false,
}: {
  language: Language
  setLanguage: (language: Language) => void
  compact?: boolean
}) {
  return (
    <div
      className={compact ? "language compact" : "language"}
      aria-label="Choose language"
    >
      <Icon name="globe" size={17} />
      {(["en", "ar", "he"] as Language[]).map((lang) => (
        <button
          key={lang}
          className={language === lang ? "active" : ""}
          onClick={() => setLanguage(lang)}
          aria-pressed={language === lang}
        >
          {lang.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

function initialsOf(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean)
  return ((parts[0]?.[0] ?? "") + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase() || "U"
}

function Login({
  language,
  setLanguage,
  onEnter,
}: {
  language: Language
  setLanguage: (language: Language) => void
  onEnter: (role: Role, employee: Employee | null) => void
}) {
  const [role, setRole] = useState<Role>("employee")
  const [loginEmail, setLoginEmail] = useState("")
  const [checking, setChecking] = useState(false)
  const [loginError, setLoginError] = useState<string | null>(null)
  const t = copy[language]
  const c = chatCopy[language]
  const needsEmail = role === "employee"
  const emailOk = isEmailValid(loginEmail)

  // Employees must sign in with a directory-verified email; the name shown
  // afterwards is the one returned by the directory.
  const proceed = async () => {
    if (!needsEmail) return onEnter(role, null)
    if (!emailOk || checking) return
    setChecking(true)
    setLoginError(null)
    try {
      const employee = await identifyEmployee(loginEmail)
      if (employee) onEnter(role, employee)
      else setLoginError(c.signInFailed)
    } catch {
      setLoginError(c.signInUnavailable)
    } finally {
      setChecking(false)
    }
  }

  const roles: { key: Role; icon: IconName; title: string; desc: string }[] = [
    { key: "employee", icon: "user", title: t.employee, desc: t.employeeDesc },
    { key: "support", icon: "headset", title: t.support, desc: t.supportDesc },
    { key: "admin", icon: "shield", title: t.admin, desc: t.adminDesc },
  ]
  return (
    <main className="login-page">
      <div className="login-top">
        <Logo />
        <LanguageSwitcher language={language} setLanguage={setLanguage} />
      </div>
      <section className="login-panel">
        <div className="login-art" aria-hidden="true">
          <div className="art-orbit orbit-one" />
          <div className="art-orbit orbit-two" />
          <div className="art-grid" />
          <PalmGrove className="login-palms" />
          <div className="art-copy">
            <span className="eyebrow">
              <Icon name="sparkles" size={15} /> Intelligent support, naturally
            </span>
            <h2>
              Technology that lets
              <br />
              your work <em>flow.</em>
            </h2>
            <p>
              A calm, connected space for every question, request, and
              resolution.
            </p>
          </div>
          <div className="floating-card card-a">
            <span className="float-icon">
              <Icon name="check" />
            </span>
            <div>
              <strong>Issue resolved</strong>
              <small>VPN access restored</small>
            </div>
          </div>
          <div className="floating-card card-b">
            <span className="avatar mini">OM</span>
            <div>
              <strong>Omar is helping</strong>
              <small>Average reply · 4 min</small>
            </div>
          </div>
          <div className="art-seal">
            <Logo compact />
          </div>
        </div>
        <div className="login-form">
          <div className="mobile-logo">
            <Logo />
          </div>
          <span className="kicker">NAKHLA DIGITAL WORKPLACE</span>
          <h1>{t.welcome}</h1>
          <p className="lead">{t.loginSubtitle}</p>
          <fieldset className="role-list">
            <legend>Select your role</legend>
            {roles.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`role-option ${role === item.key ? "selected" : ""}`}
                onClick={() => setRole(item.key)}
              >
                <span className="role-icon">
                  <Icon name={item.icon} />
                </span>
                <span className="role-copy">
                  <strong>{item.title}</strong>
                  <small>{item.desc}</small>
                </span>
                <span className="radio">
                  <span />
                </span>
              </button>
            ))}
          </fieldset>
          {needsEmail && (
            <div className="email-field login-email">
              <label htmlFor="login-email">{c.emailLabel}</label>
              <input
                id="login-email"
                type="email"
                inputMode="email"
                autoComplete="email"
                dir="ltr"
                value={loginEmail}
                placeholder={c.emailPlaceholder}
                aria-invalid={loginError !== null || (loginEmail.trim() !== "" && !emailOk)}
                disabled={checking}
                onChange={(e) => {
                  setLoginEmail(e.target.value)
                  setLoginError(null)
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.nativeEvent.isComposing) void proceed()
                }}
              />
              <small className={loginError ? "invalid" : ""} role={loginError ? "alert" : undefined}>
                {loginError ??
                  (loginEmail.trim() !== "" && !emailOk ? c.emailInvalid : c.emailHint)}
              </small>
            </div>
          )}
          <button
            className="primary-button login-button"
            disabled={checking || (needsEmail && !emailOk)}
            onClick={() => void proceed()}
          >
            {checking ? c.signingIn : t.continue}
            <Icon name="arrow" />
          </button>
          <div className="prototype-note">
            <Icon name="lock" size={14} />
            {t.prototype}
          </div>
        </div>
      </section>
      <footer className="login-footer">
        <span>© 2025 Nakhla Technologies</span>
        <span>Privacy · Accessibility · Support</span>
      </footer>
    </main>
  )
}

function StatusBadge({ status }: { status: string }) {
  const cls = status.toLowerCase().replaceAll(" ", "-")
  return (
    <span className={`badge ${cls}`}>
      <i />
      {status}
    </span>
  )
}

function Shell({
  role,
  view,
  setView,
  language,
  setLanguage,
  onLogout,
  user,
  canGoBack,
  onBack,
  children,
}: {
  role: Role
  view: View
  setView: (view: View) => void
  language: Language
  setLanguage: (language: Language) => void
  onLogout: () => void
  user: Employee | null
  canGoBack: boolean
  onBack: () => void
  children: React.ReactNode
}) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const t = copy[language]
  const c = chatCopy[language]
  const displayName =
    role === "employee" && user
      ? user.name
      : role === "admin"
        ? "Noor Al-Sayed"
        : role === "support"
          ? "Omar Khalil"
          : ""
  const initials = displayName ? initialsOf(displayName) : "U"
  const nav =
    role === "employee"
      ? [
          { id: "chat", label: t.aiAssistant, icon: "sparkles" },
          { id: "tickets", label: t.myTickets, icon: "ticket" },
        ]
      : role === "support"
        ? [
            { id: "support", label: t.overview, icon: "home" },
            { id: "tickets", label: t.assigned, icon: "ticket" },
          ]
        : [
            { id: "admin", label: t.overview, icon: "chart" },
            { id: "users", label: t.users, icon: "users" },
            { id: "reports", label: t.reports, icon: "report" },
          ]
  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="sidebar-top">
          <Logo />
          <button
            className="mobile-close"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <Icon name="x" />
          </button>
        </div>
        <nav>
          <span className="nav-label">
            {role === "employee"
              ? "PERSONAL DESK"
              : role === "support"
                ? "SUPPORT DESK"
                : "ADMINISTRATION"}
          </span>
          {nav.map((item) => (
            <button
              key={item.id}
              className={
                view === item.id || (view === "ticket" && item.id === "tickets")
                  ? "active"
                  : ""
              }
              onClick={() => {
                setView(item.id as View)
                setMobileOpen(false)
              }}
            >
              <Icon name={item.icon as IconName} />
              <span>{item.label}</span>
              {item.id === "tickets" && role !== "admin" && (
                <b>{role === "support" ? "8" : "4"}</b>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-help">
          <span>
            <Icon name="headset" />
          </span>
          <strong>Need urgent help?</strong>
          <small>Call IT at extension 4400</small>
        </div>
        <button className="profile-card" onClick={onLogout}>
          <span className="avatar">{initials}</span>
          <span>
            <strong>{displayName}</strong>
            <small>
              {role === "employee" && user
                ? user.title || user.department || t.employee
                : role === "admin"
                  ? t.admin
                  : role === "support"
                    ? t.support
                    : t.employee}
            </small>
          </span>
          <Icon name="logout" size={18} />
        </button>
      </aside>
      {mobileOpen && (
        <button
          className="scrim"
          onClick={() => setMobileOpen(false)}
          aria-label="Close navigation"
        />
      )}
      <section className="main-area">
        <header className="topbar">
          <button
            className="menu-button"
            onClick={() => setMobileOpen(true)}
            aria-label="Open navigation"
          >
            <Icon name="menu" />
          </button>
          <button
            className="back-button"
            disabled={!canGoBack}
            onClick={onBack}
            aria-label={c.back}
            title={c.back}
          >
            <Icon name="arrow" size={17} />
            <span>{c.back}</span>
          </button>
          <div className="topbar-search">
            <Icon name="search" size={18} />
            <span>Search help, tickets, or people...</span>
            <kbd>⌘ K</kbd>
          </div>
          <div className="topbar-actions">
            <LanguageSwitcher
              compact
              language={language}
              setLanguage={setLanguage}
            />
            <button className="icon-button" aria-label="Notifications">
              <Icon name="bell" />
              <i />
            </button>
            <span className="top-avatar">{initials}</span>
          </div>
        </header>
        <main className="workspace">
          <BotanicalCorner className="workspace-botanical" />
          <BotanicalCorner className="workspace-botanical workspace-botanical-low" />
          <PalmGrove className="workspace-palms" />
          {children}
        </main>
      </section>
    </div>
  )
}

function ThinkingStatus({ phases }: { phases: readonly string[] }) {
  const [step, setStep] = useState(0)
  useEffect(() => {
    const timer = setInterval(
      () => setStep((current) => Math.min(current + 1, phases.length - 1)),
      2500,
    )
    return () => clearInterval(timer)
  }, [phases.length])
  return <span>{phases[step]}</span>
}

function EmployeeChat({
  setView,
  language,
  chat,
  user,
}: {
  setView: (view: View) => void
  language: Language
  chat: HelpdeskChat
  user: Employee | null
}) {
  const t = copy[language]
  const c = chatCopy[language]
  const firstName = user?.name.split(/\s+/)[0] ?? ""
  const {
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
  } = chat
  const [input, setInput] = useState("")
  const [emailTouched, setEmailTouched] = useState(false)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const asked = messages.length > 0
  const emailInvalid = email.trim() !== "" && !isEmailValid(email)
  const showEmailError =
    emailInvalid && (emailTouched || error?.kind === "email")
  const draft = useMemo(
    () => (pendingAction ? extractDraft(pendingAction) : null),
    [pendingAction],
  )
  const draftBlocked = busy !== null || !pendingAction
  const rejectedNotice = outcome?.type === "rejected" && !pendingAction
  const createdTicketId =
    outcome?.type === "approved" ? findTicketId(outcome.result) : ""
  // Collapse the details again whenever a different draft is shown.
  const draftId = pendingAction?.action_id
  useEffect(() => setDetailsOpen(false), [draftId])

  const submit = async (text: string) => {
    setEmailTouched(true)
    if (await chat.send(text, email)) setInput("")
  }
  const decide = async (approved: boolean) => {
    if (approved) await chat.approve()
    else await chat.reject()
  }
  const errorText = (current: NonNullable<typeof error>) => {
    if (current.kind === "email") return c.emailInvalid
    const base =
      current.kind === "network"
        ? c.errNetwork
        : current.kind === "timeout"
          ? c.errTimeout
          : current.kind === "http"
            ? c.errHttp(current.status ?? 0)
            : c.errInvalid
    return current.scope === "approval" ? base + c.errApprovalSuffix : base
  }
  const emailField = user ? (
    <div className="email-field signed-in">
      <label>{c.signedInAs}</label>
      <strong dir="ltr">
        {user.name} · {user.email}
      </strong>
    </div>
  ) : (
    <div className="email-field">
      <label htmlFor="work-email">{c.emailLabel}</label>
      <input
        id="work-email"
        type="email"
        inputMode="email"
        autoComplete="email"
        dir="ltr"
        value={email}
        placeholder={c.emailPlaceholder}
        aria-invalid={showEmailError}
        aria-describedby="work-email-hint"
        onChange={(e) => setEmail(e.target.value)}
        onBlur={() => setEmailTouched(true)}
      />
      <small id="work-email-hint" className={showEmailError ? "invalid" : ""}>
        {showEmailError ? c.emailInvalid : c.emailHint}
      </small>
    </div>
  )
  const suggestions = [
    {
      icon: "lock" as IconName,
      title: "I can’t connect to the VPN",
      desc: "Troubleshoot remote access",
    },
    {
      icon: "laptop" as IconName,
      title: "My laptop is running slowly",
      desc: "Improve device performance",
    },
    {
      icon: "user" as IconName,
      title: "I need access to a system",
      desc: "Request permissions",
    },
    {
      icon: "message" as IconName,
      title: "Teams or Outlook issue",
      desc: "Fix Microsoft 365 apps",
    },
  ]
  return (
    <div className="chat-layout">
      <section className="chat-main">
        <div className="page-heading chat-heading">
          <div>
            <span className="kicker">AI-POWERED SUPPORT</span>
            <h1>{t.goodMorning.replace("{name}", firstName).replace(/[,،]\s*$/, "")}</h1>
            <p>{t.chatSubtitle}</p>
          </div>
          <button
            className="secondary-button"
            disabled={busy !== null}
            onClick={() => {
              chat.reset()
              setInput("")
            }}
          >
            <Icon name="message" size={18} />
            {t.newConversation}
          </button>
        </div>
        {!asked ? (
          <div className="assistant-welcome">
            <div className="bot-mark">
              <Icon name="sparkles" size={28} />
            </div>
            <h2>What can we solve together?</h2>
            <p>
              Describe what’s happening in your own words. I’ll search Nakhla’s
              verified knowledge base and guide you step by step.
            </p>
            {emailField}
            <div className="chat-input large">
              <input
                aria-label={t.askPlaceholder}
                placeholder={t.askPlaceholder}
                value={input}
                disabled={busy !== null}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.nativeEvent.isComposing)
                    void submit(input)
                }}
              />
              <button
                className="send-button"
                aria-label={c.send}
                disabled={busy !== null || !input.trim()}
                onClick={() => void submit(input)}
              >
                <Icon name="send" />
              </button>
            </div>
            {error?.scope === "chat" && error.kind !== "email" && (
              <div className="chat-error" role="alert">
                {errorText(error)}
              </div>
            )}
            <div className="suggestion-label">
              <span>{t.suggested}</span>
              <i />
            </div>
            <div className="suggestion-grid">
              {suggestions.map((item) => (
                <button
                  key={item.title}
                  className="suggestion-card"
                  disabled={busy !== null}
                  onClick={() => void submit(item.title)}
                >
                  <span>
                    <Icon name={item.icon} />
                  </span>
                  <div>
                    <strong>{item.title}</strong>
                    <small>{item.desc}</small>
                  </div>
                  <Icon name="chevron" size={17} />
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="conversation">
            {messages.map((message) =>
              message.role === "user" ? (
                <div key={message.id} className="message-row user-message">
                  <div>
                    <small>
                      {c.you} · {message.time.toLocaleTimeString(language, { hour: "2-digit", minute: "2-digit" })}
                    </small>
                    <p className="message-text">{message.text}</p>
                  </div>
                  <span className="avatar mini">
                    {user ? initialsOf(user.name) : (email.trim()[0] || "U").toUpperCase()}
                  </span>
                </div>
              ) : (
                <div key={message.id} className="message-row ai-message">
                  <span className="bot-avatar">
                    <Icon name="sparkles" size={18} />
                  </span>
                  <div>
                    <small>
                      {c.assistant} · {message.time.toLocaleTimeString(language, { hour: "2-digit", minute: "2-digit" })}
                    </small>
                    <div className="message-bubble">
                      <p className="message-text">{message.text}</p>
                    </div>
                  </div>
                </div>
              ),
            )}
            {busy === "chat" && (
              <div className="message-row ai-message" role="status">
                <span className="bot-avatar">
                  <Icon name="sparkles" size={18} />
                </span>
                <div>
                  <div className="message-bubble typing-indicator">
                    <i />
                    <i />
                    <i />
                    <ThinkingStatus phases={c.phases} />
                  </div>
                </div>
              </div>
            )}
            {error && error.kind !== "email" && (
              <div className="chat-error" role="alert">
                <span>{errorText(error)}</span>
                {error.scope === "chat" && failedMessage && (
                  <button
                    disabled={busy !== null}
                    onClick={() => void chat.send(failedMessage, email)}
                  >
                    {c.retry}
                  </button>
                )}
                <button onClick={chat.clearError}>{c.dismiss}</button>
              </div>
            )}
            {pendingActionInvalid && (
              <div className="chat-error" role="alert">
                {c.invalidDraft}
              </div>
            )}
            {rejectedNotice && (
              <div className="draft-cancelled" role="status">
                <span>
                  <Icon name="x" size={16} />
                </span>
                <div>
                  <strong>{c.draftCancelled}</strong>
                  <p>{c.draftCancelledBody}</p>
                </div>
                <button onClick={chat.clearOutcome}>{c.dismiss}</button>
              </div>
            )}
            {emailField}
            {draft && pendingAction && (
              <section
                className="ticket-card"
                aria-labelledby="ticket-card-title"
                aria-busy={busy === "approve" || busy === "reject"}
              >
                <header>
                  <span className="soft-icon">
                    <Icon name="ticket" size={16} />
                  </span>
                  <h3 id="ticket-card-title">{c.pendingTitle}</h3>
                  <button
                    className="ticket-card-toggle"
                    aria-expanded={detailsOpen}
                    onClick={() => setDetailsOpen((open) => !open)}
                  >
                    {detailsOpen ? c.hideDetails : c.showDetails}
                  </button>
                </header>
                <strong className="ticket-card-subject">
                  {draft.subject || draft.title || c.noDraftDetails}
                </strong>
                <div className="ticket-card-meta">
                  {draft.category && <span>{draft.category}</span>}
                  {draft.priority && (
                    <span className="priority">
                      <i />
                      {draft.priority}
                    </span>
                  )}
                  {draft.requesterEmail && (
                    <span dir="ltr">{draft.requesterEmail}</span>
                  )}
                </div>
                {(draft.description || draft.attemptedSteps.length > 0) && (
                  <p className={`ticket-card-summary${detailsOpen ? " open" : ""}`}>
                    {draft.description}
                    {draft.attemptedSteps.length > 0 &&
                      ` ${c.attemptedSteps}: ${draft.attemptedSteps.join("; ")}.`}
                  </p>
                )}
                {detailsOpen && (
                  <dl className="ticket-card-details">
                    {draft.department && (
                      <>
                        <dt>{c.department}</dt>
                        <dd>{draft.department}</dd>
                      </>
                    )}
                    {draft.kbReferences.length > 0 && (
                      <>
                        <dt>{c.kbReferences}</dt>
                        <dd>{draft.kbReferences.join(", ")}</dd>
                      </>
                    )}
                  </dl>
                )}
                {outcome?.type === "failed" && (
                  <div className="chat-error" role="alert">
                    {c.failedTitle}. {c.errApprovalSuffix.trim()}
                  </div>
                )}
                <footer>
                  <button
                    className="approve-button"
                    disabled={draftBlocked}
                    onClick={() => void decide(true)}
                  >
                    <Icon name="check" size={16} />
                    {busy === "approve" ? c.approving : c.approve}
                  </button>
                  <button
                    className="reject-button"
                    disabled={draftBlocked}
                    onClick={() => void decide(false)}
                  >
                    <Icon name="x" size={16} />
                    {busy === "reject" ? c.rejecting : c.reject}
                  </button>
                </footer>
              </section>
            )}
            {!pendingAction && createdTicketId && (
              <div className="draft-cancelled ticket-created" role="status">
                <span>
                  <Icon name="check" size={16} />
                </span>
                <div>
                  <strong>
                    {c.approvedTitle}: <span dir="ltr">{createdTicketId}</span>
                  </strong>
                  <p>{c.approvedBody}</p>
                </div>
                <button onClick={chat.clearOutcome}>{c.dismiss}</button>
              </div>
            )}
            <div className="chat-input docked">
              <input
                aria-label={c.followUp}
                placeholder={c.followUp}
                value={input}
                disabled={busy !== null}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.nativeEvent.isComposing)
                    void submit(input)
                }}
              />
              <button
                className="send-button"
                aria-label={c.send}
                disabled={busy !== null || !input.trim()}
                onClick={() => void submit(input)}
              >
                <Icon name="send" />
              </button>
            </div>
          </div>
        )}
      </section>
      <aside className="chat-side">
        <DiagnosticsPanel diagnostics={diagnostics} t={c} />
        <div className="side-card ticket-glance">
          <div className="card-title">
            <div>
              <span className="soft-icon">
                <Icon name="ticket" />
              </span>
              <h3>{t.openTicket}</h3>
            </div>
            <button onClick={() => setView("tickets")}>{t.viewAll}</button>
          </div>
          <strong>VPN connection drops frequently</strong>
          <div className="ticket-meta">
            <span>#IT-2841</span>
            <StatusBadge status="In progress" />
          </div>
          <div className="assignee">
            <span className="avatar mini">OK</span>
            <div>
              <small>Assigned to</small>
              <strong>Omar Khalil</strong>
            </div>
            <span className="response-time">
              <Icon name="clock" size={14} />~ 12 min
            </span>
          </div>
        </div>
        <div className="side-card tips-card">
          <span className="soft-icon peach">
            <Icon name="sparkles" />
          </span>
          <h3>For faster help</h3>
          <p>
            Include the app name, error message, and when the issue started.
          </p>
          <a href="#guide">
            How to describe an issue <Icon name="arrow" size={15} />
          </a>
        </div>
        <div className="service-status">
          <i />
          <span>
            <strong>All systems operational</strong>
            <small>Last checked 2 minutes ago</small>
          </span>
        </div>
      </aside>
    </div>
  )
}

function TicketsPage({
  setView,
  supportMode = false,
  live,
  hasEmail,
  onRefresh,
}: {
  setView: (view: View) => void
  supportMode?: boolean
  live: ReturnType<typeof useTickets>
  hasEmail: boolean
  onRefresh: () => void
}) {
  const [filter, setFilter] = useState("All")
  // Employees see their real tickets; the support queue is still demo data.
  const useLive = !supportMode
  const rows: TicketRow[] = useLive ? (live.list?.tickets ?? []) : tickets
  const stats = live.list ? computeStats(live.list) : null
  const statCards: string[][] = useLive
    ? [
        [stats ? String(stats.total) : "–", "Total tickets", "ticket"],
        [stats ? String(stats.open) : "–", "Open", "clock"],
        [stats ? String(stats.resolved) : "–", "Resolved", "check"],
        [stats ? String(stats.closed) : "–", "Closed", "x"],
      ]
    : [
        ["8", "Total tickets", "ticket"],
        ["1", "In progress", "clock"],
        ["1", "Waiting on you", "message"],
        ["2", "Resolved", "check"],
      ]
  const tabs = useLive
    ? ["All", "Open", "Resolved", "Closed"]
    : ["All", "In progress", "Waiting", "Resolved"]
  const visible =
    filter === "All" ? rows : rows.filter((ticket) => ticket.status === filter)
  const emptyMessage = !useLive
    ? ""
    : !hasEmail
      ? "Enter your work email in the chat to see your tickets."
      : live.status === "loading"
        ? "Loading your tickets…"
        : live.status === "error"
          ? "Could not load your tickets. Check that the email is a verified Nakhla address and try again."
          : "You have no tickets yet."
  return (
    <div>
      <div className="page-heading">
        <div>
          <span className="kicker">
            {supportMode ? "SUPPORT QUEUE" : "REQUEST CENTER"}
          </span>
          <h1>{supportMode ? "Assigned tickets" : "My tickets"}</h1>
          <p>
            {supportMode
              ? "Prioritize your queue and keep requests moving."
              : "Track updates and revisit your previous IT requests."}
          </p>
        </div>
        {!supportMode && (
          <button className="primary-button">
            <Icon name="sparkles" size={17} />
            Ask AI Assistant
          </button>
        )}
      </div>
      <div className="ticket-stats">
        {statCards.map(([value, label, icon], index) => (
          <div className="mini-stat" key={label}>
            <span className={`stat-icon stat-${index}`}>
              <Icon name={icon as IconName} />
            </span>
            <div>
              <strong>{value}</strong>
              <small>{label}</small>
            </div>
          </div>
        ))}
      </div>
      <div className="list-card">
        <div className="list-tools">
          <div className="search-field">
            <Icon name="search" size={18} />
            <input
              aria-label="Search tickets"
              placeholder="Search by title or ticket ID"
            />
          </div>
          <div className="filter-tabs">
            {tabs.map((item) => (
              <button
                key={item}
                className={filter === item ? "active" : ""}
                onClick={() => setFilter(item)}
              >
                {item}
              </button>
            ))}
          </div>
          <button className="filter-button" onClick={useLive ? onRefresh : undefined}>
            <Icon name="filter" size={17} />
            {useLive ? "Refresh" : "Filters"}
          </button>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Ticket</th>
                <th>Category</th>
                <th>Priority</th>
                <th>Status</th>
                <th>Last update</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {useLive && visible.length === 0 && (
                <tr>
                  <td colSpan={6}>{emptyMessage}</td>
                </tr>
              )}
              {visible.map((ticket) => (
                <tr
                  key={ticket.id}
                  onClick={useLive ? undefined : () => setView("ticket")}
                  tabIndex={0}
                >
                  <td>
                    <span className="ticket-id">{ticket.id}</span>
                    <strong>{ticket.title}</strong>
                  </td>
                  <td>{ticket.category}</td>
                  <td>
                    <span
                      className={`priority ${ticket.priority.toLowerCase()}`}
                    >
                      <i />
                      {ticket.priority}
                    </span>
                  </td>
                  <td>
                    <StatusBadge status={ticket.status} />
                  </td>
                  <td>{ticket.date}</td>
                  <td>
                    <button
                      className="row-action"
                      aria-label={`Open ${ticket.id}`}
                    >
                      <Icon name="chevron" size={17} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function TicketDetail({
  setView,
  supportMode = false,
}: {
  setView: (view: View) => void
  supportMode?: boolean
}) {
  const [status, setStatus] = useState("In progress")
  return (
    <div>
      <button
        className="back-link"
        onClick={() => setView(supportMode ? "support" : "tickets")}
      >
        <Icon name="arrow" size={17} />
        Back to tickets
      </button>
      <div className="detail-header">
        <div>
          <div className="detail-meta">
            <span>IT-2841</span>
            <StatusBadge status={status} />
            <span className="priority high">
              <i />
              High priority
            </span>
          </div>
          <h1>VPN connection drops frequently</h1>
          <p>Created today at 9:42 AM · Network & Connectivity</p>
        </div>
        <button className="secondary-button">
          <Icon name="more" />
          More actions
        </button>
      </div>
      <div className="detail-grid">
        <section className="detail-main">
          <div className="conversation-card">
            <div className="section-title">
              <h2>Conversation</h2>
              <span>3 messages</span>
            </div>
            <div className="thread-message">
              <span className="avatar">LA</span>
              <div>
                <div className="thread-head">
                  <strong>Layla Ahmed</strong>
                  <time>Today, 9:42 AM</time>
                </div>
                <p>
                  My VPN keeps disconnecting every few minutes while I’m working
                  from home. I’ve restarted the laptop and my router, but the
                  problem continues.
                </p>
                <div className="attachment">
                  <Icon name="paperclip" size={17} />
                  <span>
                    <strong>vpn-error.png</strong>
                    <small>PNG · 284 KB</small>
                  </span>
                  <Icon name="download" size={17} />
                </div>
              </div>
            </div>
            <div className="thread-message support-reply">
              <span className="avatar green">OK</span>
              <div>
                <div className="thread-head">
                  <strong>
                    Omar Khalil <em>IT Support</em>
                  </strong>
                  <time>Today, 9:58 AM</time>
                </div>
                <p>
                  Hi Layla, I’m looking into this. I can see a few interrupted
                  sessions. Could you confirm whether the issue happens on both
                  Wi-Fi and a mobile hotspot?
                </p>
              </div>
            </div>
            <div className="thread-message">
              <span className="avatar">LA</span>
              <div>
                <div className="thread-head">
                  <strong>Layla Ahmed</strong>
                  <time>Today, 10:06 AM</time>
                </div>
                <p>
                  Yes, I tested with my mobile hotspot and it disconnected there
                  too.
                </p>
              </div>
            </div>
            <div className="reply-box">
              <textarea aria-label="Reply" placeholder="Write a reply..." />
              <div>
                <button>
                  <Icon name="paperclip" size={18} />
                  Attach file
                </button>
                <button className="primary-button">
                  Send reply
                  <Icon name="send" size={17} />
                </button>
              </div>
            </div>
          </div>
          {supportMode && (
            <div className="internal-note">
              <div className="section-title">
                <h2>
                  <Icon name="lock" size={17} />
                  Internal note
                </h2>
                <span>Only visible to IT</span>
              </div>
              <textarea
                aria-label="Internal note"
                placeholder="Add technical context for the support team..."
              />
              <button className="secondary-button">Add note</button>
            </div>
          )}
        </section>
        <aside className="detail-side">
          <div className="detail-card">
            <h3>Ticket details</h3>
            <dl>
              <div>
                <dt>Assigned technician</dt>
                <dd>
                  <span className="avatar mini green">OK</span>
                  <span>
                    <strong>Omar Khalil</strong>
                    <small>Network Specialist</small>
                  </span>
                </dd>
              </div>
              <div>
                <dt>Priority</dt>
                <dd>
                  <span className="priority high">
                    <i />
                    High
                  </span>
                </dd>
              </div>
              <div>
                <dt>Category</dt>
                <dd>Network & Connectivity</dd>
              </div>
              <div>
                <dt>Device</dt>
                <dd>Nakhla-MBP-042</dd>
              </div>
            </dl>
            {supportMode && (
              <div className="status-control">
                <label htmlFor="status">Update status</label>
                <select
                  id="status"
                  value={status}
                  onChange={(e) => setStatus(e.target.value)}
                >
                  <option>Open</option>
                  <option>In progress</option>
                  <option>Waiting</option>
                  <option>Resolved</option>
                </select>
                <small>Prototype only — changes are not saved.</small>
              </div>
            )}
          </div>
          <div className="detail-card">
            <h3>Progress</h3>
            <div className="timeline">
              <div className="done">
                <i>
                  <Icon name="check" size={12} />
                </i>
                <span>
                  <strong>Open</strong>
                  <small>Today, 9:42 AM</small>
                </span>
              </div>
              <div className="current">
                <i />
                <span>
                  <strong>In progress</strong>
                  <small>Today, 9:51 AM</small>
                </span>
              </div>
              <div>
                <i />
                <span>
                  <strong>Resolved</strong>
                  <small>Pending</small>
                </span>
              </div>
            </div>
          </div>
          <div className="detail-card related">
            <span className="soft-icon">
              <Icon name="book" />
            </span>
            <div>
              <small>RELATED ARTICLE</small>
              <strong>VPN connectivity guide</strong>
              <a href="#article">
                Read article <Icon name="arrow" size={14} />
              </a>
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}

function SupportDashboard({ setView }: { setView: (view: View) => void }) {
  return (
    <div>
      <div className="page-heading">
        <div>
          <span className="kicker">THURSDAY, MAY 15</span>
          <h1>Good morning, Omar</h1>
          <p>Here’s what needs your attention today.</p>
        </div>
        <div className="availability">
          <i />
          Available for assignments
          <Icon name="chevron" size={16} />
        </div>
      </div>
      <div className="support-stats">
        {[
          ["8", "Assigned to me", "2 new since yesterday", "ticket"],
          ["3", "Due today", "1 approaching SLA", "clock"],
          ["14", "Resolved this week", "18% from last week", "check"],
          ["4.8", "Satisfaction score", "From 23 responses", "trend"],
        ].map(([value, label, note, icon], index) => (
          <div className="dashboard-stat" key={label}>
            <div>
              <span className={`stat-icon stat-${index}`}>
                <Icon name={icon as IconName} />
              </span>
              <span className={index === 2 ? "positive" : ""}>{note}</span>
            </div>
            <strong>{value}</strong>
            <small>{label}</small>
          </div>
        ))}
      </div>
      <div className="dashboard-grid">
        <section className="list-card queue-card">
          <div className="section-title">
            <div>
              <h2>Priority queue</h2>
              <p>Tickets that need your attention first</p>
            </div>
            <button onClick={() => setView("tickets")}>
              View all <Icon name="arrow" size={15} />
            </button>
          </div>
          <div className="queue-list">
            {tickets.slice(0, 3).map((ticket, index) => (
              <button key={ticket.id} onClick={() => setView("ticket")}>
                <span className={`queue-bar q-${index}`} />
                <div>
                  <span>
                    <b>{ticket.id}</b>
                    <small>{ticket.category}</small>
                  </span>
                  <strong>{ticket.title}</strong>
                  <span className="queue-bottom">
                    <span
                      className={`priority ${ticket.priority.toLowerCase()}`}
                    >
                      <i />
                      {ticket.priority}
                    </span>
                    <span>
                      <Icon name="clock" size={14} />
                      {index === 0
                        ? "SLA in 42 min"
                        : index === 1
                          ? "Updated 18 min ago"
                          : "Waiting 1h"}
                    </span>
                  </span>
                </div>
                <Icon name="chevron" size={18} />
              </button>
            ))}
          </div>
        </section>
        <aside className="today-card">
          <div className="section-title">
            <div>
              <h2>Today’s activity</h2>
              <p>Thursday, May 15</p>
            </div>
          </div>
          <div className="activity-ring">
            <svg viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="49" />
              <circle className="progress" cx="60" cy="60" r="49" />
            </svg>
            <div>
              <strong>6</strong>
              <small>resolved</small>
            </div>
          </div>
          <div className="activity-legend">
            <span>
              <i className="green-dot" />
              <b>6</b>
              <small>Resolved</small>
            </span>
            <span>
              <i className="peach-dot" />
              <b>4</b>
              <small>In progress</small>
            </span>
            <span>
              <i className="gray-dot" />
              <b>2</b>
              <small>Waiting</small>
            </span>
          </div>
          <div className="daily-goal">
            <span>
              <Icon name="trend" size={17} />
              Daily goal
            </span>
            <strong>6 of 8 tickets</strong>
            <i>
              <span />
            </i>
          </div>
        </aside>
      </div>
    </div>
  )
}

function MiniChart() {
  return (
    <div className="chart-area">
      <div className="y-labels">
        <span>120</span>
        <span>90</span>
        <span>60</span>
        <span>30</span>
        <span>0</span>
      </div>
      <svg
        viewBox="0 0 700 220"
        preserveAspectRatio="none"
        role="img"
        aria-label="Ticket volume trend over seven months"
      >
        <defs>
          <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#19806b" stopOpacity=".24" />
            <stop offset="1" stopColor="#19806b" stopOpacity="0" />
          </linearGradient>
        </defs>
        <g className="gridlines">
          <path d="M0 10H700M0 60H700M0 110H700M0 160H700M0 210H700" />
        </g>
        <path
          className="chart-fill"
          d="M0 164 C55 145 70 128 115 137 S185 98 230 113 S300 70 350 84 S420 46 465 58 S540 30 585 48 S650 18 700 28 L700 220 L0 220Z"
        />
        <path
          className="chart-line"
          d="M0 164 C55 145 70 128 115 137 S185 98 230 113 S300 70 350 84 S420 46 465 58 S540 30 585 48 S650 18 700 28"
        />
      </svg>
      <div className="x-labels">
        <span>Nov</span>
        <span>Dec</span>
        <span>Jan</span>
        <span>Feb</span>
        <span>Mar</span>
        <span>Apr</span>
        <span>May</span>
      </div>
    </div>
  )
}

function AdminDashboard({ setView }: { setView: (view: View) => void }) {
  return (
    <div>
      <div className="page-heading">
        <div>
          <span className="kicker">DOMAIN OVERVIEW</span>
          <h1>IT operations at a glance</h1>
          <p>Performance and service health across Nakhla.</p>
        </div>
        <div className="date-button">
          <Icon name="clock" size={17} />
          Last 30 days
          <Icon name="chevron" size={15} />
        </div>
      </div>
      <div className="admin-stats">
        {[
          ["284", "Total tickets", "+12%", "ticket"],
          ["32", "Open tickets", "-8%", "clock"],
          ["2.4h", "Avg. resolution", "-18%", "trend"],
          ["94%", "SLA compliance", "+3%", "shield"],
        ].map(([value, label, delta, icon], index) => (
          <div className="admin-stat" key={label}>
            <div>
              <span className={`stat-icon stat-${index}`}>
                <Icon name={icon as IconName} />
              </span>
              <span className={index === 0 ? "neutral-delta" : "good-delta"}>
                {delta}
              </span>
            </div>
            <strong>{value}</strong>
            <small>{label}</small>
            <p>
              {index === 0
                ? "vs. previous period"
                : index === 1
                  ? "4 high priority"
                  : index === 2
                    ? "Target: under 4h"
                    : "Target: 90%"}
            </p>
          </div>
        ))}
      </div>
      <div className="analytics-grid">
        <section className="chart-card">
          <div className="section-title">
            <div>
              <h2>Ticket volume</h2>
              <p>Created vs. resolved tickets</p>
            </div>
            <div className="chart-legend">
              <span>
                <i />
                Created
              </span>
              <span>
                <i />
                Resolved
              </span>
            </div>
          </div>
          <MiniChart />
        </section>
        <section className="category-card">
          <div className="section-title">
            <div>
              <h2>Tickets by category</h2>
              <p>Top request areas</p>
            </div>
            <button>
              <Icon name="more" />
            </button>
          </div>
          <div className="donut-wrap">
            <div className="donut">
              <div>
                <strong>284</strong>
                <small>Total</small>
              </div>
            </div>
            <div className="donut-legend">
              {[
                ["Software", "34%", "c1"],
                ["Access", "26%", "c2"],
                ["Network", "22%", "c3"],
                ["Hardware", "18%", "c4"],
              ].map(([name, value, cls]) => (
                <div key={name}>
                  <span>
                    <i className={cls} />
                    {name}
                  </span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
      <div className="admin-bottom">
        <section className="team-card">
          <div className="section-title">
            <div>
              <h2>Team performance</h2>
              <p>Support specialist workload</p>
            </div>
            <button onClick={() => setView("users")}>
              Manage team <Icon name="arrow" size={15} />
            </button>
          </div>
          {[
            ["OK", "Omar Khalil", "Network Specialist", "18", "2.1h", "96%"],
            ["SM", "Sara Mansour", "Application Support", "15", "2.8h", "93%"],
            ["YA", "Yousef Ali", "Systems Specialist", "12", "3.2h", "91%"],
          ].map((person, index) => (
            <div className="team-row" key={person[1]}>
              <span className={`avatar avatar-${index}`}>{person[0]}</span>
              <div>
                <strong>{person[1]}</strong>
                <small>{person[2]}</small>
              </div>
              <span>
                <small>Resolved</small>
                <strong>{person[3]}</strong>
              </span>
              <span>
                <small>Avg. time</small>
                <strong>{person[4]}</strong>
              </span>
              <span>
                <small>SLA</small>
                <strong>{person[5]}</strong>
              </span>
            </div>
          ))}
        </section>
        <section className="health-card">
          <div className="section-title">
            <div>
              <h2>Service health</h2>
              <p>Current system status</p>
            </div>
          </div>
          {[
            "Microsoft 365",
            "VPN & Remote Access",
            "Network Services",
            "ERP Systems",
          ].map((service) => (
            <div className="health-row" key={service}>
              <span>
                <i />
                {service}
              </span>
              <strong>Operational</strong>
            </div>
          ))}
          <div className="uptime">
            <span>30-day uptime</span>
            <strong>99.97%</strong>
          </div>
        </section>
      </div>
    </div>
  )
}

function UsersPage() {
  const people = [
    ["OK", "Omar Khalil", "omar.khalil@nakhla.com", "IT Support", "Active"],
    ["SM", "Sara Mansour", "sara.mansour@nakhla.com", "IT Support", "Active"],
    ["LA", "Layla Ahmed", "layla.ahmed@nakhla.com", "Employee", "Active"],
    ["YA", "Yousef Ali", "yousef.ali@nakhla.com", "IT Support", "Away"],
    ["MK", "Mariam Karam", "mariam.karam@nakhla.com", "Employee", "Active"],
  ]
  return (
    <div>
      <div className="page-heading">
        <div>
          <span className="kicker">ACCESS & ROLES</span>
          <h1>User management</h1>
          <p>Review prototype user roles and workspace access.</p>
        </div>
        <button className="primary-button">
          <Icon name="user" size={17} />
          Invite user
        </button>
      </div>
      <div className="ticket-stats">
        <div className="mini-stat">
          <span className="stat-icon stat-0">
            <Icon name="users" />
          </span>
          <div>
            <strong>148</strong>
            <small>Total users</small>
          </div>
        </div>
        <div className="mini-stat">
          <span className="stat-icon stat-1">
            <Icon name="headset" />
          </span>
          <div>
            <strong>12</strong>
            <small>IT specialists</small>
          </div>
        </div>
        <div className="mini-stat">
          <span className="stat-icon stat-2">
            <Icon name="shield" />
          </span>
          <div>
            <strong>4</strong>
            <small>Administrators</small>
          </div>
        </div>
      </div>
      <div className="list-card">
        <div className="list-tools">
          <div className="search-field">
            <Icon name="search" size={18} />
            <input
              aria-label="Search users"
              placeholder="Search people or email"
            />
          </div>
          <button className="filter-button">
            <Icon name="filter" size={17} />
            All roles
          </button>
        </div>
        <div className="user-list">
          {people.map((person, index) => (
            <div className="user-row" key={person[2]}>
              <span className={`avatar avatar-${index % 3}`}>{person[0]}</span>
              <span>
                <strong>{person[1]}</strong>
                <small>{person[2]}</small>
              </span>
              <span className="role-pill">{person[3]}</span>
              <span className={`presence ${person[4].toLowerCase()}`}>
                <i />
                {person[4]}
              </span>
              <button aria-label={`Actions for ${person[1]}`}>
                <Icon name="more" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function ReportsPage() {
  return (
    <div>
      <div className="page-heading">
        <div>
          <span className="kicker">INSIGHTS & EXPORTS</span>
          <h1>Reports</h1>
          <p>Ready-to-review service desk summaries and performance data.</p>
        </div>
        <button className="primary-button">
          <Icon name="download" size={17} />
          Export summary
        </button>
      </div>
      <div className="report-grid">
        {[
          [
            "Monthly service review",
            "Ticket trends, SLA performance, and satisfaction.",
            "Updated today",
            "chart",
          ],
          [
            "Support team performance",
            "Resolution time and workload by specialist.",
            "Updated yesterday",
            "users",
          ],
          [
            "Category analysis",
            "Volume and root causes by request category.",
            "Updated May 13",
            "report",
          ],
          [
            "SLA compliance",
            "Response and resolution targets by priority.",
            "Updated May 12",
            "shield",
          ],
        ].map(([title, desc, update, icon], index) => (
          <article className="report-card" key={title}>
            <span className={`report-icon r-${index}`}>
              <Icon name={icon as IconName} />
            </span>
            <div>
              <h2>{title}</h2>
              <p>{desc}</p>
              <small>{update}</small>
            </div>
            <button aria-label={`Open ${title}`}>
              <Icon name="arrow" />
            </button>
          </article>
        ))}
      </div>
      <section className="chart-card report-chart">
        <div className="section-title">
          <div>
            <h2>Monthly resolution trend</h2>
            <p>Resolved tickets over the last seven months</p>
          </div>
          <div className="date-button">
            All categories
            <Icon name="chevron" size={14} />
          </div>
        </div>
        <MiniChart />
      </section>
    </div>
  )
}

export default function App() {
  const [language, setLanguage] = useState<Language>("en")
  const [role, setRole] = useState<Role | null>(null)
  const [view, setViewState] = useState<View>("chat")
  const [viewHistory, setViewHistory] = useState<View[]>([])
  const [user, setUser] = useState<Employee | null>(null)
  const chat = useHelpdeskChat()
  const liveTickets = useTickets(chat.email, chat.outcome)
  useEffect(() => {
    document.documentElement.lang = language
    document.documentElement.dir = language === "en" ? "ltr" : "rtl"
  }, [language])

  const initialView = useMemo(
    () =>
      ({
        employee: "chat",
        support: "support",
        admin: "admin",
      }) as Record<Role, View>,
    [],
  )
  const setView = (next: View) => {
    if (next === view) return
    setViewHistory((history) => [...history, view])
    setViewState(next)
  }
  const goBack = () => {
    const previous = viewHistory[viewHistory.length - 1]
    if (!previous) return
    setViewHistory((history) => history.slice(0, -1))
    setViewState(previous)
  }
  const enter = (nextRole: Role, employee: Employee | null) => {
    chat.signOut()
    setUser(employee)
    if (employee) chat.setEmail(employee.email)
    setRole(nextRole)
    setViewHistory([])
    setViewState(initialView[nextRole])
  }
  // Signing out wipes the session, messages and identity so the next person
  // never sees the previous employee's name or conversation.
  const logout = () => {
    chat.signOut()
    setUser(null)
    setRole(null)
    setViewHistory([])
  }

  if (!role)
    return (
      <LocalizedContent key={language} language={language}>
        <Login language={language} setLanguage={setLanguage} onEnter={enter} />
      </LocalizedContent>
    )

  let screen: React.ReactNode
  if (view === "chat")
    screen = <EmployeeChat setView={setView} language={language} chat={chat} user={user} />
  else if (view === "tickets")
    screen = (
      <TicketsPage
        setView={setView}
        supportMode={role === "support"}
        live={liveTickets}
        hasEmail={chat.email.trim() !== ""}
        onRefresh={() => void liveTickets.refresh()}
      />
    )
  else if (view === "ticket")
    screen = <TicketDetail setView={setView} supportMode={role === "support"} />
  else if (view === "support") screen = <SupportDashboard setView={setView} />
  else if (view === "users") screen = <UsersPage />
  else if (view === "reports") screen = <ReportsPage />
  else screen = <AdminDashboard setView={setView} />

  return (
    <LocalizedContent key={language} language={language}>
      <Shell
        role={role}
        view={view}
        setView={setView}
        language={language}
        setLanguage={setLanguage}
        onLogout={logout}
        user={user}
        canGoBack={viewHistory.length > 0}
        onBack={goBack}
      >
        {screen}
      </Shell>
    </LocalizedContent>
  )
}

import { useLayoutEffect, useRef, type ReactNode } from "react"

export type AppLanguage = "en" | "ar" | "he"

type Translation = { ar: string; he: string }

const catalog: Record<string, Translation> = {
  "IT HELPDESK": { ar: "مكتب دعم تقنية المعلومات", he: "מוקד תמיכת IT" },
  "Choose language": { ar: "اختر اللغة", he: "בחירת שפה" },
  "Intelligent support, naturally": {
    ar: "دعم ذكي بكل بساطة",
    he: "תמיכה חכמה, באופן טבעי",
  },
  "Technology that lets": { ar: "تقنية تجعل", he: "טכנולוגיה שמאפשרת" },
  "your work": { ar: "عملك", he: "לעבודה שלך" },
  "flow.": { ar: "ينساب بسلاسة.", he: "לזרום." },
  "A calm, connected space for every question, request, and resolution.": {
    ar: "مساحة هادئة ومترابطة لكل سؤال وطلب وحل.",
    he: "מרחב רגוע ומחובר לכל שאלה, בקשה ופתרון.",
  },
  "Issue resolved": { ar: "تم حل المشكلة", he: "הבעיה נפתרה" },
  "VPN access restored": {
    ar: "تمت استعادة اتصال VPN",
    he: "הגישה ל־VPN שוחזרה",
  },
  "Omar is helping": { ar: "عمر يساعدك الآن", he: "עומר מסייע כעת" },
  "Average reply · 4 min": {
    ar: "متوسط الرد · 4 دقائق",
    he: "זמן תגובה ממוצע · 4 דקות",
  },
  "NAKHLA DIGITAL WORKPLACE": {
    ar: "بيئة العمل الرقمية لنخلة",
    he: "סביבת העבודה הדיגיטלית של Nakhla",
  },
  "Select your role": { ar: "اختر دورك", he: "בחירת תפקיד" },
  "Privacy · Accessibility · Support": {
    ar: "الخصوصية · إمكانية الوصول · الدعم",
    he: "פרטיות · נגישות · תמיכה",
  },
  "Close navigation": { ar: "إغلاق قائمة التنقل", he: "סגירת הניווט" },
  "Open navigation": { ar: "فتح قائمة التنقل", he: "פתיחת הניווט" },
  "PERSONAL DESK": { ar: "مكتب الموظف", he: "המרחב האישי" },
  "SUPPORT DESK": { ar: "مكتب الدعم", he: "מוקד התמיכה" },
  ADMINISTRATION: { ar: "الإدارة", he: "ניהול" },
  "Need urgent help?": {
    ar: "هل تحتاج مساعدة عاجلة؟",
    he: "זקוקים לעזרה דחופה?",
  },
  "Call IT at extension 4400": {
    ar: "اتصل بالدعم التقني على التحويلة 4400",
    he: "התקשרו לתמיכת IT בשלוחה 4400",
  },
  "Search help, tickets, or people...": {
    ar: "ابحث في المساعدة أو التذاكر أو الأشخاص...",
    he: "חיפוש בעזרה, בכרטיסים או באנשים...",
  },
  Notifications: { ar: "الإشعارات", he: "התראות" },
  "AI-POWERED SUPPORT": {
    ar: "دعم مدعوم بالذكاء الاصطناعي",
    he: "תמיכה מבוססת בינה מלאכותית",
  },
  "What can we solve together?": {
    ar: "ما الذي يمكننا حله معاً؟",
    he: "מה נוכל לפתור יחד?",
  },
  "Describe what’s happening in your own words. I’ll search Nakhla’s verified knowledge base and guide you step by step.":
    {
      ar: "صِف ما يحدث بكلماتك. سأبحث في قاعدة معرفة نخلة الموثوقة وأرشدك خطوة بخطوة.",
      he: "תארו במילים שלכם מה קורה. אחפש במאגר הידע המאומת של Nakhla ואנחה אתכם שלב אחר שלב.",
    },
  "Attach file": { ar: "إرفاق ملف", he: "צירוף קובץ" },
  "Send message": { ar: "إرسال الرسالة", he: "שליחת הודעה" },
  "You · just now": { ar: "أنت · الآن", he: "את/ה · עכשיו" },
  "My VPN keeps disconnecting every few minutes when I work from home.": {
    ar: "يتكرر انقطاع اتصال VPN كل بضع دقائق أثناء عملي من المنزل.",
    he: "חיבור ה־VPN שלי מתנתק כל כמה דקות כשאני עובד מהבית.",
  },
  "Nakhla Assistant · just now": {
    ar: "مساعد نخلة · الآن",
    he: "העוזר של Nakhla · עכשיו",
  },
  "I found a likely fix. First, let’s refresh your VPN profile:": {
    ar: "وجدت حلاً محتملاً. لنحدّث أولاً ملف تعريف VPN:",
    he: "מצאתי פתרון אפשרי. תחילה נרענן את פרופיל ה־VPN:",
  },
  Open: { ar: "مفتوحة", he: "פתוח" },
  "from your menu bar.": { ar: "من شريط القوائم.", he: "משורת התפריטים." },
  "Select Settings → Refresh Connection.": {
    ar: "اختر الإعدادات ← تحديث الاتصال.",
    he: "בחרו הגדרות ← רענון חיבור.",
  },
  "Reconnect using your Nakhla credentials.": {
    ar: "أعد الاتصال باستخدام بيانات اعتماد نخلة.",
    he: "התחברו מחדש באמצעות פרטי ההזדהות של Nakhla.",
  },
  "If it still disconnects, I recommend creating a support ticket so our network team can check your connection logs.":
    {
      ar: "إذا استمر الانقطاع، أوصي بإنشاء تذكرة دعم ليفحص فريق الشبكات سجلات الاتصال.",
      he: "אם החיבור ממשיך להתנתק, מומלץ ליצור כרטיס תמיכה כדי שצוות הרשת יבדוק את יומני החיבור.",
    },
  "· VPN connectivity guide": {
    ar: "· دليل اتصال VPN",
    he: "· מדריך קישוריות VPN",
  },
  "Updated 6 days ago": {
    ar: "تم التحديث قبل 6 أيام",
    he: "עודכן לפני 6 ימים",
  },
  "Ticket draft cancelled": {
    ar: "تم إلغاء مسودة التذكرة",
    he: "טיוטת הכרטיס בוטלה",
  },
  "No ticket was created. You can continue this conversation.": {
    ar: "لم يتم إنشاء أي تذكرة. يمكنك متابعة المحادثة.",
    he: "לא נוצר כרטיס. אפשר להמשיך בשיחה.",
  },
  "Review again": { ar: "مراجعة مجدداً", he: "בדיקה חוזרת" },
  "Ask a follow-up question...": {
    ar: "اطرح سؤالاً إضافياً...",
    he: "שאלו שאלת המשך...",
  },
  "I can’t connect to the VPN": {
    ar: "لا أستطيع الاتصال بشبكة VPN",
    he: "לא ניתן להתחבר ל־VPN",
  },
  "Troubleshoot remote access": {
    ar: "استكشاف مشكلات الوصول عن بُعد",
    he: "פתרון בעיות גישה מרחוק",
  },
  "My laptop is running slowly": {
    ar: "الحاسوب المحمول يعمل ببطء",
    he: "המחשב הנייד פועל באיטיות",
  },
  "Improve device performance": {
    ar: "تحسين أداء الجهاز",
    he: "שיפור ביצועי המכשיר",
  },
  "I need access to a system": {
    ar: "أحتاج إلى الوصول إلى نظام",
    he: "נדרשת לי גישה למערכת",
  },
  "Request permissions": { ar: "طلب صلاحيات", he: "בקשת הרשאות" },
  "Teams or Outlook issue": {
    ar: "مشكلة في Teams أو Outlook",
    he: "בעיה ב־Teams או ב־Outlook",
  },
  "Fix Microsoft 365 apps": {
    ar: "إصلاح تطبيقات Microsoft 365",
    he: "תיקון יישומי Microsoft 365",
  },
  "AI TICKET DRAFT": { ar: "مسودة تذكرة ذكية", he: "טיוטת כרטיס חכמה" },
  "Ready for your review": { ar: "جاهزة لمراجعتك", he: "מוכן לבדיקה שלך" },
  "Check the proposed ticket before deciding. Nothing is submitted until you explicitly approve it.":
    {
      ar: "راجع التذكرة المقترحة قبل اتخاذ القرار. لن يتم إرسال شيء حتى توافق صراحةً.",
      he: "בדקו את הכרטיס המוצע לפני ההחלטה. דבר לא יישלח ללא אישור מפורש.",
    },
  "VPN connection drops frequently": {
    ar: "اتصال VPN ينقطع بشكل متكرر",
    he: "חיבור ה־VPN מתנתק לעיתים קרובות",
  },
  "Network & Connectivity": { ar: "الشبكة والاتصال", he: "רשת וקישוריות" },
  High: { ar: "عالية", he: "גבוהה" },
  Medium: { ar: "متوسطة", he: "בינונית" },
  Low: { ar: "منخفضة", he: "נמוכה" },
  "3 relevant details included": {
    ar: "تم تضمين 3 تفاصيل ذات صلة",
    he: "נכללו 3 פרטים רלוונטיים",
  },
  "Review draft": { ar: "مراجعة المسودة", he: "בדיקת הטיוטה" },
  "Reject without creating": { ar: "رفض دون إنشاء", he: "דחייה ללא יצירה" },
  "No ticket has been created": {
    ar: "لم يتم إنشاء أي تذكرة",
    he: "לא נוצר כרטיס",
  },
  "Assigned to": { ar: "مُعيّنة إلى", he: "הוקצה אל" },
  "For faster help": {
    ar: "للحصول على مساعدة أسرع",
    he: "לקבלת עזרה מהירה יותר",
  },
  "Include the app name, error message, and when the issue started.": {
    ar: "اذكر اسم التطبيق ورسالة الخطأ ووقت بدء المشكلة.",
    he: "ציינו את שם היישום, הודעת השגיאה ומתי הבעיה התחילה.",
  },
  "How to describe an issue": { ar: "كيفية وصف المشكلة", he: "איך לתאר בעיה" },
  "All systems operational": {
    ar: "جميع الأنظمة تعمل",
    he: "כל המערכות פעילות",
  },
  "Last checked 2 minutes ago": {
    ar: "آخر فحص قبل دقيقتين",
    he: "נבדק לאחרונה לפני 2 דקות",
  },
  "REVIEW BEFORE SUBMITTING": {
    ar: "راجع قبل الإرسال",
    he: "בדיקה לפני שליחה",
  },
  "Review ticket draft": {
    ar: "مراجعة مسودة التذكرة",
    he: "בדיקת טיוטת הכרטיס",
  },
  "Close ticket draft": { ar: "إغلاق مسودة التذكرة", he: "סגירת טיוטת הכרטיס" },
  "Nothing has been created yet.": {
    ar: "لم يتم إنشاء أي شيء بعد.",
    he: "עדיין לא נוצר דבר.",
  },
  "Review the proposed details below, then explicitly approve or reject this draft.":
    {
      ar: "راجع التفاصيل المقترحة أدناه، ثم وافق على المسودة أو ارفضها صراحةً.",
      he: "בדקו את הפרטים המוצעים למטה, ואז אשרו או דחו את הטיוטה במפורש.",
    },
  "Proposed title": { ar: "العنوان المقترح", he: "כותרת מוצעת" },
  Description: { ar: "الوصف", he: "תיאור" },
  "VPN disconnects every few minutes while working remotely. Restarting the laptop and router did not resolve the issue, and the behavior also occurs on a mobile hotspot.":
    {
      ar: "ينقطع اتصال VPN كل بضع دقائق أثناء العمل عن بُعد. لم تؤدِ إعادة تشغيل الحاسوب والموجّه إلى حل المشكلة، كما تحدث عند استخدام نقطة اتصال الهاتف.",
      he: "חיבור ה־VPN מתנתק כל כמה דקות בעבודה מרחוק. הפעלה מחדש של המחשב והנתב לא פתרה את הבעיה, והיא מתרחשת גם בחיבור לנקודה חמה סלולרית.",
    },
  Category: { ar: "الفئة", he: "קטגוריה" },
  Priority: { ar: "الأولوية", he: "עדיפות" },
  "Relevant evidence": { ar: "الأدلة ذات الصلة", he: "ראיות רלוונטיות" },
  "Included from this AI conversation": {
    ar: "مُضمّنة من هذه المحادثة الذكية",
    he: "נכלל משיחת הבינה המלאכותית",
  },
  "3 items": { ar: "3 عناصر", he: "3 פריטים" },
  "User report": { ar: "بلاغ المستخدم", he: "דיווח משתמש" },
  "Disconnects every few minutes from home": {
    ar: "ينقطع كل بضع دقائق من المنزل",
    he: "מתנתק כל כמה דקות מהבית",
  },
  "Troubleshooting attempted": {
    ar: "محاولات استكشاف المشكلة",
    he: "ניסיונות פתרון",
  },
  "Laptop and router restarted": {
    ar: "تمت إعادة تشغيل الحاسوب والموجّه",
    he: "המחשב והנתב הופעלו מחדש",
  },
  "Knowledge-base reference": {
    ar: "مرجع قاعدة المعرفة",
    he: "מקור ממאגר הידע",
  },
  "KB-1042 · VPN connectivity guide": {
    ar: "KB-1042 · دليل اتصال VPN",
    he: "KB-1042 · מדריך קישוריות VPN",
  },
  "Reject draft": { ar: "رفض المسودة", he: "דחיית הטיוטה" },
  "Approve ticket": { ar: "الموافقة على التذكرة", he: "אישור הכרטיס" },
  "DEMO CONFIRMATION": { ar: "تأكيد تجريبي", he: "אישור הדגמה" },
  "Ticket approved": { ar: "تمت الموافقة على التذكرة", he: "הכרטיס אושר" },
  "This prototype has not contacted a backend or created a real ticket. The ID and tracking link below are demonstration data.":
    {
      ar: "لم يتصل هذا النموذج بأي خادم خلفي ولم ينشئ تذكرة حقيقية. المعرّف ورابط التتبع أدناه بيانات تجريبية.",
      he: "אב־טיפוס זה לא פנה למערכת אחורית ולא יצר כרטיס אמיתי. המזהה וקישור המעקב שלהלן הם נתוני הדגמה.",
    },
  "Demo ticket ID": { ar: "معرّف تذكرة تجريبي", he: "מזהה כרטיס לדוגמה" },
  "Demo tracking link": { ar: "رابط تتبع تجريبي", he: "קישור מעקב לדוגמה" },
  "Connect this action to the FastAPI backend to generate and track real tickets.":
    {
      ar: "اربط هذا الإجراء بخادم FastAPI لإنشاء تذاكر حقيقية وتتبعها.",
      he: "חברו פעולה זו למערכת FastAPI כדי ליצור כרטיסים אמיתיים ולעקוב אחריהם.",
    },
  Done: { ar: "تم", he: "סיום" },
  "SUPPORT QUEUE": { ar: "قائمة انتظار الدعم", he: "תור התמיכה" },
  "REQUEST CENTER": { ar: "مركز الطلبات", he: "מרכז הבקשות" },
  "Assigned tickets": { ar: "التذاكر المعيّنة", he: "כרטיסים שהוקצו" },
  "My tickets": { ar: "تذاكري", he: "הכרטיסים שלי" },
  "Prioritize your queue and keep requests moving.": {
    ar: "رتّب قائمة الانتظار وحافظ على تقدم الطلبات.",
    he: "תעדפו את התור ושמרו על התקדמות הבקשות.",
  },
  "Track updates and revisit your previous IT requests.": {
    ar: "تابع التحديثات وراجع طلباتك التقنية السابقة.",
    he: "עקבו אחר עדכונים וחזרו לבקשות IT קודמות.",
  },
  "Ask AI Assistant": { ar: "اسأل المساعد الذكي", he: "שאלו את העוזר החכם" },
  "Total tickets": { ar: "إجمالي التذاكر", he: "סך הכרטיסים" },
  "In progress": { ar: "قيد المعالجة", he: "בטיפול" },
  "Waiting on you": { ar: "بانتظار ردك", he: "ממתין לתגובתך" },
  Waiting: { ar: "بانتظار الرد", he: "בהמתנה" },
  Resolved: { ar: "محلولة", he: "נפתר" },
  All: { ar: "الكل", he: "הכול" },
  Filters: { ar: "عوامل التصفية", he: "מסננים" },
  "Search tickets": { ar: "البحث في التذاكر", he: "חיפוש כרטיסים" },
  "Search by title or ticket ID": {
    ar: "ابحث بالعنوان أو معرّف التذكرة",
    he: "חיפוש לפי כותרת או מזהה כרטיס",
  },
  Ticket: { ar: "التذكرة", he: "כרטיס" },
  Status: { ar: "الحالة", he: "סטטוס" },
  "Last update": { ar: "آخر تحديث", he: "עדכון אחרון" },
  Actions: { ar: "الإجراءات", he: "פעולות" },
  Network: { ar: "الشبكات", he: "רשת" },
  Software: { ar: "البرمجيات", he: "תוכנה" },
  Access: { ar: "الوصول", he: "גישה" },
  Hardware: { ar: "الأجهزة", he: "חומרה" },
  "Microsoft Teams audio not working": {
    ar: "الصوت لا يعمل في Microsoft Teams",
    he: "השמע ב־Microsoft Teams אינו פועל",
  },
  "Request access to Finance shared drive": {
    ar: "طلب وصول إلى مجلد المالية المشترك",
    he: "בקשת גישה לכונן המשותף של הכספים",
  },
  "Laptop battery draining quickly": {
    ar: "بطارية الحاسوب تنفد بسرعة",
    he: "סוללת המחשב מתרוקנת במהירות",
  },
  "Today, 9:42 AM": { ar: "اليوم، 9:42 ص", he: "היום, 9:42" },
  "Yesterday, 3:15 PM": { ar: "أمس، 3:15 م", he: "אתמול, 15:15" },
  "May 12, 11:30 AM": { ar: "12 مايو، 11:30 ص", he: "12 במאי, 11:30" },
  "May 10, 2:08 PM": { ar: "10 مايو، 2:08 م", he: "10 במאי, 14:08" },
  "Back to tickets": { ar: "العودة إلى التذاكر", he: "חזרה לכרטיסים" },
  "High priority": { ar: "أولوية عالية", he: "עדיפות גבוהה" },
  "Created today at 9:42 AM · Network & Connectivity": {
    ar: "أُنشئت اليوم الساعة 9:42 ص · الشبكة والاتصال",
    he: "נוצר היום ב־9:42 · רשת וקישוריות",
  },
  "More actions": { ar: "إجراءات إضافية", he: "פעולות נוספות" },
  Conversation: { ar: "المحادثة", he: "שיחה" },
  "3 messages": { ar: "3 رسائل", he: "3 הודעות" },
  "Today, 9:58 AM": { ar: "اليوم، 9:58 ص", he: "היום, 9:58" },
  "Today, 10:06 AM": { ar: "اليوم، 10:06 ص", he: "היום, 10:06" },
  "Today, 9:51 AM": { ar: "اليوم، 9:51 ص", he: "היום, 9:51" },
  "My VPN keeps disconnecting every few minutes while I’m working from home. I’ve restarted the laptop and my router, but the problem continues.":
    {
      ar: "يتكرر انقطاع اتصال VPN كل بضع دقائق أثناء عملي من المنزل. أعدت تشغيل الحاسوب والموجّه، لكن المشكلة مستمرة.",
      he: "חיבור ה־VPN מתנתק כל כמה דקות כשאני עובד מהבית. הפעלתי מחדש את המחשב והנתב, אך הבעיה נמשכת.",
    },
  "Hi Layla, I’m looking into this. I can see a few interrupted sessions. Could you confirm whether the issue happens on both Wi-Fi and a mobile hotspot?":
    {
      ar: "مرحباً ليلى، أتحقق من المشكلة الآن. أرى عدة جلسات منقطعة. هل تحدث المشكلة على شبكة Wi‑Fi ونقطة اتصال الهاتف معاً؟",
      he: "היי ליילה, אני בודק את זה. אני רואה כמה הפעלות שנקטעו. האם הבעיה מתרחשת גם ב־Wi‑Fi וגם בנקודה חמה סלולרית?",
    },
  "Yes, I tested with my mobile hotspot and it disconnected there too.": {
    ar: "نعم، جرّبت نقطة اتصال الهاتف وانقطع الاتصال هناك أيضاً.",
    he: "כן, בדקתי גם בנקודה החמה והחיבור התנתק גם שם.",
  },
  Reply: { ar: "الرد", he: "תגובה" },
  "Write a reply...": { ar: "اكتب رداً...", he: "כתיבת תגובה..." },
  "Send reply": { ar: "إرسال الرد", he: "שליחת תגובה" },
  "Internal note": { ar: "ملاحظة داخلية", he: "הערה פנימית" },
  "Only visible to IT": {
    ar: "مرئية لفريق الدعم التقني فقط",
    he: "גלוי רק לצוות IT",
  },
  "Add technical context for the support team...": {
    ar: "أضف سياقاً تقنياً لفريق الدعم...",
    he: "הוספת הקשר טכני לצוות התמיכה...",
  },
  "Add note": { ar: "إضافة ملاحظة", he: "הוספת הערה" },
  "Ticket details": { ar: "تفاصيل التذكرة", he: "פרטי הכרטיס" },
  "Assigned technician": { ar: "الفني المعيّن", he: "טכנאי מטפל" },
  "Network Specialist": { ar: "أخصائي شبكات", he: "מומחה רשתות" },
  Device: { ar: "الجهاز", he: "מכשיר" },
  "Update status": { ar: "تحديث الحالة", he: "עדכון סטטוס" },
  "Prototype only — changes are not saved.": {
    ar: "نموذج تجريبي فقط — لا يتم حفظ التغييرات.",
    he: "אב־טיפוס בלבד — השינויים אינם נשמרים.",
  },
  Progress: { ar: "التقدم", he: "התקדמות" },
  Pending: { ar: "قيد الانتظار", he: "ממתין" },
  "RELATED ARTICLE": { ar: "مقالة ذات صلة", he: "מאמר קשור" },
  "VPN connectivity guide": { ar: "دليل اتصال VPN", he: "מדריך קישוריות VPN" },
  "Read article": { ar: "قراءة المقالة", he: "קריאת המאמר" },
  "THURSDAY, MAY 15": { ar: "الخميس، 15 مايو", he: "יום חמישי, 15 במאי" },
  "Good morning, Omar": { ar: "صباح الخير، عمر", he: "בוקר טוב, עומר" },
  "Here’s what needs your attention today.": {
    ar: "إليك ما يحتاج اهتمامك اليوم.",
    he: "אלה הנושאים שדורשים את תשומת לבך היום.",
  },
  "Available for assignments": {
    ar: "متاح لاستلام المهام",
    he: "זמין להקצאות",
  },
  "Assigned to me": { ar: "مُعيّنة إليّ", he: "הוקצו לי" },
  "2 new since yesterday": { ar: "2 جديدتان منذ أمس", he: "2 חדשים מאתמול" },
  "Due today": { ar: "مستحقة اليوم", he: "לטיפול היום" },
  "1 approaching SLA": { ar: "1 تقترب من مهلة SLA", he: "1 מתקרב ליעד SLA" },
  "Resolved this week": { ar: "تم حلها هذا الأسبوع", he: "נפתרו השבוע" },
  "18% from last week": {
    ar: "18% مقارنة بالأسبوع الماضي",
    he: "18% לעומת השבוע שעבר",
  },
  "Satisfaction score": { ar: "درجة الرضا", he: "ציון שביעות רצון" },
  "From 23 responses": { ar: "من 23 استجابة", he: "מתוך 23 תגובות" },
  "Priority queue": { ar: "قائمة الأولويات", he: "תור עדיפויות" },
  "Tickets that need your attention first": {
    ar: "التذاكر التي تحتاج اهتمامك أولاً",
    he: "כרטיסים שדורשים טיפול ראשון",
  },
  "View all": { ar: "عرض الكل", he: "הצג הכול" },
  "SLA in 42 min": { ar: "مهلة SLA خلال 42 دقيقة", he: "SLA בעוד 42 דקות" },
  "Updated 18 min ago": {
    ar: "تم التحديث قبل 18 دقيقة",
    he: "עודכן לפני 18 דקות",
  },
  "Waiting 1h": { ar: "بانتظار الرد منذ ساعة", he: "בהמתנה שעה" },
  "Today’s activity": { ar: "نشاط اليوم", he: "הפעילות היום" },
  "Thursday, May 15": { ar: "الخميس، 15 مايو", he: "יום חמישי, 15 במאי" },
  resolved: { ar: "محلولة", he: "נפתרו" },
  "Daily goal": { ar: "الهدف اليومي", he: "יעד יומי" },
  "6 of 8 tickets": { ar: "6 من 8 تذاكر", he: "6 מתוך 8 כרטיסים" },
  "Ticket volume trend over seven months": {
    ar: "اتجاه حجم التذاكر خلال سبعة أشهر",
    he: "מגמת נפח כרטיסים לאורך שבעה חודשים",
  },
  Nov: { ar: "نوف", he: "נוב׳" },
  Dec: { ar: "ديس", he: "דצמ׳" },
  Jan: { ar: "ينا", he: "ינו׳" },
  Feb: { ar: "فبر", he: "פבר׳" },
  Mar: { ar: "مار", he: "מרץ" },
  Apr: { ar: "أبر", he: "אפר׳" },
  May: { ar: "مايو", he: "מאי" },
  "DOMAIN OVERVIEW": { ar: "نظرة عامة على النطاق", he: "סקירת הדומיין" },
  "IT operations at a glance": {
    ar: "نظرة سريعة على عمليات تقنية المعلومات",
    he: "פעילות IT במבט אחד",
  },
  "Performance and service health across Nakhla.": {
    ar: "الأداء وصحة الخدمات في نخلة.",
    he: "ביצועים ותקינות שירותים ב־Nakhla.",
  },
  "Last 30 days": { ar: "آخر 30 يوماً", he: "30 הימים האחרונים" },
  "Open tickets": { ar: "التذاكر المفتوحة", he: "כרטיסים פתוחים" },
  "Avg. resolution": { ar: "متوسط وقت الحل", he: "זמן פתרון ממוצע" },
  "SLA compliance": { ar: "الالتزام باتفاقية SLA", he: "עמידה ב־SLA" },
  "vs. previous period": {
    ar: "مقارنة بالفترة السابقة",
    he: "לעומת התקופה הקודמת",
  },
  "4 high priority": { ar: "4 ذات أولوية عالية", he: "4 בעדיפות גבוהה" },
  "Target: under 4h": { ar: "الهدف: أقل من 4 ساعات", he: "יעד: פחות מ־4 שעות" },
  "Target: 90%": { ar: "الهدف: 90%", he: "יעד: 90%" },
  "Ticket volume": { ar: "حجم التذاكر", he: "נפח כרטיסים" },
  "Created vs. resolved tickets": {
    ar: "التذاكر المنشأة مقابل المحلولة",
    he: "כרטיסים שנוצרו לעומת שנפתרו",
  },
  Created: { ar: "منشأة", he: "נוצרו" },
  "Tickets by category": { ar: "التذاكر حسب الفئة", he: "כרטיסים לפי קטגוריה" },
  "Top request areas": {
    ar: "أبرز مجالات الطلبات",
    he: "תחומי הבקשות המובילים",
  },
  Total: { ar: "الإجمالي", he: "סה״כ" },
  "Team performance": { ar: "أداء الفريق", he: "ביצועי הצוות" },
  "Support specialist workload": {
    ar: "عبء عمل أخصائيي الدعم",
    he: "עומס מומחי התמיכה",
  },
  "Manage team": { ar: "إدارة الفريق", he: "ניהול הצוות" },
  "Application Support": { ar: "دعم التطبيقات", he: "תמיכת יישומים" },
  "Systems Specialist": { ar: "أخصائي أنظمة", he: "מומחה מערכות" },
  "Avg. time": { ar: "متوسط الوقت", he: "זמן ממוצע" },
  "Service health": { ar: "صحة الخدمات", he: "תקינות השירותים" },
  "Current system status": {
    ar: "حالة الأنظمة الحالية",
    he: "מצב המערכות הנוכחי",
  },
  "VPN & Remote Access": { ar: "VPN والوصول عن بُعد", he: "VPN וגישה מרחוק" },
  "Network Services": { ar: "خدمات الشبكة", he: "שירותי רשת" },
  "ERP Systems": { ar: "أنظمة ERP", he: "מערכות ERP" },
  Operational: { ar: "تعمل بشكل طبيعي", he: "פעיל" },
  "30-day uptime": { ar: "وقت التشغيل خلال 30 يوماً", he: "זמינות ב־30 יום" },
  "ACCESS & ROLES": { ar: "الوصول والأدوار", he: "גישה ותפקידים" },
  "User management": { ar: "إدارة المستخدمين", he: "ניהול משתמשים" },
  "Review prototype user roles and workspace access.": {
    ar: "راجع أدوار المستخدمين والوصول إلى مساحات العمل في النموذج.",
    he: "בדקו תפקידי משתמשים וגישה לסביבות העבודה באב־הטיפוס.",
  },
  "Invite user": { ar: "دعوة مستخدم", he: "הזמנת משתמש" },
  "Total users": { ar: "إجمالي المستخدمين", he: "סך המשתמשים" },
  "IT specialists": { ar: "أخصائيو تقنية المعلومات", he: "מומחי IT" },
  Administrators: { ar: "المديرون", he: "מנהלים" },
  "Search users": { ar: "البحث عن المستخدمين", he: "חיפוש משתמשים" },
  "Search people or email": {
    ar: "ابحث عن شخص أو بريد إلكتروني",
    he: "חיפוש אנשים או דוא״ל",
  },
  "All roles": { ar: "جميع الأدوار", he: "כל התפקידים" },
  Active: { ar: "نشط", he: "פעיל" },
  Away: { ar: "غير متاح", he: "לא זמין" },
  Employee: { ar: "موظف", he: "עובד" },
  "IT Support": { ar: "الدعم التقني", he: "תמיכת IT" },
  "INSIGHTS & EXPORTS": { ar: "التحليلات والتصدير", he: "תובנות וייצוא" },
  Reports: { ar: "التقارير", he: "דוחות" },
  "Ready-to-review service desk summaries and performance data.": {
    ar: "ملخصات مكتب الخدمة وبيانات الأداء جاهزة للمراجعة.",
    he: "סיכומי מוקד השירות ונתוני הביצועים מוכנים לבדיקה.",
  },
  "Export summary": { ar: "تصدير الملخص", he: "ייצוא סיכום" },
  "Monthly service review": {
    ar: "مراجعة الخدمة الشهرية",
    he: "סקירת שירות חודשית",
  },
  "Ticket trends, SLA performance, and satisfaction.": {
    ar: "اتجاهات التذاكر وأداء SLA ومستوى الرضا.",
    he: "מגמות כרטיסים, ביצועי SLA ושביעות רצון.",
  },
  "Updated today": { ar: "تم التحديث اليوم", he: "עודכן היום" },
  "Support team performance": {
    ar: "أداء فريق الدعم",
    he: "ביצועי צוות התמיכה",
  },
  "Resolution time and workload by specialist.": {
    ar: "وقت الحل وعبء العمل حسب الأخصائي.",
    he: "זמן פתרון ועומס לפי מומחה.",
  },
  "Updated yesterday": { ar: "تم التحديث أمس", he: "עודכן אתמול" },
  "Category analysis": { ar: "تحليل الفئات", he: "ניתוח קטגוריות" },
  "Volume and root causes by request category.": {
    ar: "الحجم والأسباب الجذرية حسب فئة الطلب.",
    he: "נפח וסיבות שורש לפי קטגוריית בקשה.",
  },
  "Updated May 13": { ar: "تم التحديث في 13 مايو", he: "עודכן ב־13 במאי" },
  "Response and resolution targets by priority.": {
    ar: "أهداف الاستجابة والحل حسب الأولوية.",
    he: "יעדי תגובה ופתרון לפי עדיפות.",
  },
  "Updated May 12": { ar: "تم التحديث في 12 مايو", he: "עודכן ב־12 במאי" },
  "Monthly resolution trend": {
    ar: "اتجاه الحل الشهري",
    he: "מגמת פתרון חודשית",
  },
  "Resolved tickets over the last seven months": {
    ar: "التذاكر المحلولة خلال الأشهر السبعة الماضية",
    he: "כרטיסים שנפתרו בשבעת החודשים האחרונים",
  },
  "All categories": { ar: "جميع الفئات", he: "כל הקטגוריות" },
  "Layla Ahmed": { ar: "ليلى أحمد", he: "ליילה אחמד" },
  "Omar Khalil": { ar: "عمر خليل", he: "עומר חליל" },
  "Noor Al-Sayed": { ar: "نور السيد", he: "נור אל־סייד" },
  "Sara Mansour": { ar: "سارة منصور", he: "שרה מנסור" },
  "Yousef Ali": { ar: "يوسف علي", he: "יוסף עלי" },
  "Mariam Karam": { ar: "مريم كرم", he: "מרים כרם" },
  "2.4h": { ar: "2.4 س", he: "2.4 ש׳" },
  "2.1h": { ar: "2.1 س", he: "2.1 ש׳" },
  "2.8h": { ar: "2.8 س", he: "2.8 ש׳" },
  "3.2h": { ar: "3.2 س", he: "3.2 ש׳" },
}

function translate(value: string, language: Exclude<AppLanguage, "en">) {
  const direct = catalog[value]?.[language]
  if (direct) return direct

  const openTicket = value.match(/^Open (IT-\d+)$/)
  if (openTicket)
    return language === "ar" ? `فتح ${openTicket[1]}` : `פתיחת ${openTicket[1]}`

  const actionsFor = value.match(/^Actions for (.+)$/)
  if (actionsFor)
    return language === "ar"
      ? `إجراءات خاصة بـ ${actionsFor[1]}`
      : `פעולות עבור ${actionsFor[1]}`

  const openReport = value.match(/^Open (.+)$/)
  if (openReport)
    return language === "ar"
      ? `فتح ${catalog[openReport[1]]?.ar ?? openReport[1]}`
      : `פתיחת ${catalog[openReport[1]]?.he ?? openReport[1]}`

  return value
}

function localizeNode(root: HTMLElement, language: Exclude<AppLanguage, "en">) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let node = walker.nextNode()
  while (node) {
    const raw = node.textContent ?? ""
    const trimmed = raw.trim()
    if (trimmed) {
      const localized = translate(trimmed, language)
      if (localized !== trimmed)
        node.textContent = raw.replace(trimmed, localized)
    }
    node = walker.nextNode()
  }

  root
    .querySelectorAll<HTMLElement>("[placeholder], [aria-label], [title]")
    .forEach((element) => {
      for (const attribute of ["placeholder", "aria-label", "title"]) {
        const value = element.getAttribute(attribute)
        if (value) {
          const localized = translate(value, language)
          if (localized !== value) element.setAttribute(attribute, localized)
        }
      }
    })
}

export function LocalizedContent({
  language,
  children,
}: {
  language: AppLanguage
  children: ReactNode
}) {
  const rootRef = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const root = rootRef.current
    if (!root || language === "en") return

    const apply = () => localizeNode(root, language)
    apply()
    const observer = new MutationObserver(apply)
    observer.observe(root, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ["placeholder", "aria-label", "title"],
    })
    return () => observer.disconnect()
  }, [language])

  return (
    <div
      ref={rootRef}
      className="localized-root"
      dir={language === "en" ? "ltr" : "rtl"}
      lang={language}
    >
      {children}
    </div>
  )
}

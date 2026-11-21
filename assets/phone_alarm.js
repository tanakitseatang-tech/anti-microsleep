// Phone alarm launcher: every way a web page can try to set an alarm / timer on the driver's phone.
// Used by driver.html (rest picker) and alarm_test.html (try each way on a real phone).
//
// Limits (no web page can get past these):
//  - Android: Chrome may refuse intents to the Clock app (it only opens apps that accept links).
//    Each Android way carries S.browser_fallback_url = this page + "#alarmfail=<way>", so when Chrome
//    refuses, the page learns it and shows the next way.
//  - iPhone: Apple lets no website set an alarm. The only automatic way is a Shortcuts shortcut
//    ("พักรถ", made once on the phone); clock-timer:// / clock-alarm:// only open the Clock app.
//  - Opening another app needs a tap: each way must start from a button press.
// The way that worked last time (the page was hidden = the other app opened) is remembered.
(function () {
  const SHORTCUT_NAME = "พักรถ";
  const MSG = "หมดเวลาพัก กลับไปขับต่อ";
  const KEY_OK = "ams_alarm_way";            // localStorage: way that opened the clock last time

  const ua = navigator.userAgent || "";
  const isAndroid = /Android/i.test(ua);
  const isIOS = /iPhone|iPad|iPod/i.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const platform = isAndroid ? "android" : isIOS ? "ios" : "other";

  const pad = (n) => String(n).padStart(2, "0");
  const hhmm = (d) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const utcStamp = (d) => d.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
  const fallback = (way) => encodeURIComponent(location.href.split("#")[0] + "#alarmfail=" + way);
  const enc = encodeURIComponent;

  function intent(action, extras, way, pkg) {
    return `intent:#Intent;action=${action};${pkg ? `package=${pkg};` : ""}${extras}S.browser_fallback_url=${fallback(way)};end`;
  }
  const timerExtras = (min) => `i.android.intent.extra.alarm.LENGTH=${min * 60};S.android.intent.extra.alarm.MESSAGE=${enc(MSG)};B.android.intent.extra.alarm.SKIP_UI=false;`;
  const alarmExtras = (d) => `i.android.intent.extra.alarm.HOUR=${d.getHours()};i.android.intent.extra.alarm.MINUTES=${d.getMinutes()};S.android.intent.extra.alarm.MESSAGE=${enc(MSG)};B.android.intent.extra.alarm.SKIP_UI=false;`;

  function icsBlobUrl(end) {
    const start = new Date(end.getTime() - 60e3);
    const ics = [
      "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Anti-Microsleep//Rest//TH", "METHOD:PUBLISH",
      "BEGIN:VEVENT", `UID:ams-rest-${end.getTime()}@anti-microsleep`, `DTSTAMP:${utcStamp(new Date())}`,
      `DTSTART:${utcStamp(start)}`, `DTEND:${utcStamp(end)}`, `SUMMARY:⏰ ${MSG}`,
      "BEGIN:VALARM", "ACTION:DISPLAY", `DESCRIPTION:${MSG}`, "TRIGGER;RELATED=END:PT0M", "END:VALARM",
      "END:VEVENT", "END:VCALENDAR",
    ].join("\r\n");
    return URL.createObjectURL(new Blob([ics], { type: "text/calendar;charset=utf-8" }));
  }

  // Each way: id, label (Thai), platforms, url(min, end) -> string, note.
  const WAYS = [
    // ---- Android: Clock app (Google Clock, Samsung Clock, ...)
    { id: "a_timer", os: ["android"], label: "⏱️ ตั้งตัวจับเวลาในแอปนาฬิกา",
      url: (m) => intent("android.intent.action.SET_TIMER", timerExtras(m), "a_timer") },
    { id: "a_alarm", os: ["android"], label: "⏰ ตั้งปลุกในแอปนาฬิกา",
      url: (m, d) => intent("android.intent.action.SET_ALARM", alarmExtras(d), "a_alarm") },
    { id: "a_timer_g", os: ["android"], label: "⏱️ ตัวจับเวลา (นาฬิกา Google)",
      url: (m) => intent("android.intent.action.SET_TIMER", timerExtras(m), "a_timer_g", "com.google.android.deskclock") },
    { id: "a_alarm_g", os: ["android"], label: "⏰ ตั้งปลุก (นาฬิกา Google)",
      url: (m, d) => intent("android.intent.action.SET_ALARM", alarmExtras(d), "a_alarm_g", "com.google.android.deskclock") },
    { id: "a_alarm_s", os: ["android"], label: "⏰ ตั้งปลุก (นาฬิกา Samsung)",
      url: (m, d) => intent("android.intent.action.SET_ALARM", alarmExtras(d), "a_alarm_s", "com.sec.android.app.clockpackage") },
    { id: "a_show", os: ["android"], label: "📂 เปิดหน้าปลุกในแอปนาฬิกา (ตั้งเวลาเอง)",
      url: () => intent("android.intent.action.SHOW_ALARMS", "", "a_show") },
    { id: "a_gcal", os: ["android", "other"], label: "📅 ลง Google ปฏิทิน (แจ้งเตือนตามการตั้งค่าปฏิทิน)",
      url: (m, d) => `https://calendar.google.com/calendar/render?action=TEMPLATE&text=${enc("⏰ " + MSG)}&dates=${utcStamp(d)}/${utcStamp(new Date(d.getTime() + 5 * 60e3))}&details=${enc("Anti-Microsleep: หมดเวลาพักรถ")}` },
    // ---- iPhone
    { id: "i_shortcut", os: ["ios"], label: `⏱️ ตั้งตัวจับเวลาอัตโนมัติ (คำสั่งลัด "${SHORTCUT_NAME}")`,
      url: (m) => `shortcuts://run-shortcut?name=${enc(SHORTCUT_NAME)}&input=text&text=${m}`,
      note: "ต้องสร้างคำสั่งลัดครั้งแรกครั้งเดียว (กดดูวิธี)" },
    { id: "i_timer", os: ["ios"], label: "⏱️ เปิดหน้าตัวจับเวลาในแอปนาฬิกา (ตั้งเอง)",
      url: () => "clock-timer://" },
    { id: "i_alarm", os: ["ios"], label: "⏰ เปิดหน้าปลุกในแอปนาฬิกา (ตั้งเอง)",
      url: () => "clock-alarm://" },
    // ---- both: calendar file with a reminder at the end of the rest
    { id: "ics", os: ["ios", "android", "other"], label: "📅 เพิ่มลงปฏิทินพร้อมแจ้งเตือน (.ics)",
      url: (m, d) => icsBlobUrl(d), download: "rest-alarm.ics" },
  ];

  function waysFor(os = platform) {
    const list = WAYS.filter((w) => w.os.includes(os));
    let best = null;
    try { best = localStorage.getItem(KEY_OK); } catch (e) {}
    list.sort((a, b) => (b.id === best) - (a.id === best));
    return list;
  }

  // Start one way. Must be called from a tap. onResult({ way, ok }) when known.
  function launch(way, minutes, endDate, onResult) {
    const url = way.url(minutes, endDate);
    let done = false;
    const finish = (ok) => {
      if (done) return; done = true;
      document.removeEventListener("visibilitychange", onHide);
      window.removeEventListener("hashchange", onHash);
      if (ok) { try { localStorage.setItem(KEY_OK, way.id); } catch (e) {} }
      if (onResult) onResult({ way, ok });
    };
    const onHide = () => { if (document.hidden) finish(true); };
    const onHash = () => { if (location.hash.startsWith("#alarmfail=")) { history.replaceState(null, "", location.href.split("#")[0]); finish(false); } };
    document.addEventListener("visibilitychange", onHide);
    window.addEventListener("hashchange", onHash);
    if (way.download) {
      const a = document.createElement("a");
      a.href = url; a.download = way.download; document.body.appendChild(a); a.click(); a.remove();
    } else {
      location.href = url;
    }
    // still here after 2.5 s = it did not open (a calendar file opens over the page: unknown -> null)
    setTimeout(() => finish(document.hidden ? true : way.download ? null : false), 2500);
  }

  window.PhoneAlarm = { platform, isAndroid, isIOS, SHORTCUT_NAME, WAYS, waysFor, launch, hhmm };
})();

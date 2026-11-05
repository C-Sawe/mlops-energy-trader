/**
 * The US equity session: is it open now, and when does that next change?
 *
 * No requirement ID covers this. It is a reading aid beside FR-20's status: the
 * newest bar only moves after a close, so the dashboard should show where the
 * market is in its day. It runs on the viewer's clock, needs no backend, and
 * so stays true when the dashboard is showing example data (IR-07). Nothing
 * that trades or trains reads it.
 *
 * Unlike the backend's schedule, which just finds no new bar on a holiday,
 * this tells a person the market is open, so it has to know the holidays.
 * The dates are from Alpaca's /v2/calendar, read 2026-11-05. Past the end of
 * 2027 the lists run out and only regular weekday hours apply: extend them.
 */
const MARKET_TZ = "America/New_York";
const OPEN_MINUTE = 9 * 60 + 30;
const CLOSE_MINUTE = 16 * 60;
const EARLY_CLOSE_MINUTE = 13 * 60;

const HOLIDAYS = new Set([
  "2026-11-26", "2026-12-25",
  "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
  "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
]);
const EARLY_CLOSES = new Set(["2026-11-27", "2026-12-24", "2027-11-26"]);

const nyFormat = new Intl.DateTimeFormat("en-US", {
  timeZone: MARKET_TZ,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZoneName: "short",
});

function nyParts(instant) {
  const p = Object.fromEntries(nyFormat.formatToParts(instant).map((x) => [x.type, x.value]));
  return {
    date: `${p.year}-${p.month}-${p.day}`,
    minute: Number(p.hour) * 60 + Number(p.minute),
    time: `${p.hour}:${p.minute}`,
    zone: p.timeZoneName,
  };
}

const dayUtc = (date) => new Date(`${date}T00:00:00Z`);
const weekdayName = (date) =>
  dayUtc(date).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
const hhmm = (minute) =>
  `${String(Math.floor(minute / 60)).padStart(2, "0")}:${String(minute % 60).padStart(2, "0")}`;

/** The minute this date's session closes, or null if it has no session. */
function closeMinute(date) {
  const weekday = dayUtc(date).getUTCDay();
  if (weekday === 0 || weekday === 6 || HOLIDAYS.has(date)) return null;
  return EARLY_CLOSES.has(date) ? EARLY_CLOSE_MINUTE : CLOSE_MINUTE;
}

function nextSessionDate(date) {
  const day = dayUtc(date);
  do {
    day.setUTCDate(day.getUTCDate() + 1);
  } while (closeMinute(day.toISOString().slice(0, 10)) === null);
  return day.toISOString().slice(0, 10);
}

/** The instant a New York wall-clock time occurs. New York's offset is read
 * for that date, so a daylight-saving change between now and then is honoured.
 * Only used for session times, well clear of the 02:00 changeover. */
function nyInstant(date, minute) {
  const asUtc = dayUtc(date).getTime() + minute * 60_000;
  const seen = nyParts(new Date(asUtc));
  const seenAsUtc = dayUtc(seen.date).getTime() + seen.minute * 60_000;
  return new Date(asUtc + (asUtc - seenAsUtc));
}

/**
 * Pure, so the session boundaries can be tested without rendering.
 * `next` is the next open (when closed) or the coming close (when open).
 */
export function marketStatus(now = new Date()) {
  const ny = nyParts(now);
  const close = closeMinute(ny.date);
  const open = close !== null && ny.minute >= OPEN_MINUTE && ny.minute < close;
  const beforeOpenToday = close !== null && ny.minute < OPEN_MINUTE;

  const nextDate = open || beforeOpenToday ? ny.date : nextSessionDate(ny.date);
  const nextMinute = open ? close : OPEN_MINUTE;

  return {
    open,
    holiday: HOLIDAYS.has(ny.date),
    nyTime: ny.time,
    nyZone: ny.zone,
    next: {
      at: nyInstant(nextDate, nextMinute),
      weekday: weekdayName(nextDate),
      time: hhmm(nextMinute),
    },
  };
}

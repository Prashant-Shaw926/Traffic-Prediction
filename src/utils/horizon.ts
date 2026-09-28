const MIN_MINUTES = 60
const MAX_MINUTES = 120

export function toDateTimeLocal(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`
}

export function defaultWindow(): { start: string; end: string } {
  return { start: '2017-06-15T10:00', end: '2017-06-15T12:00' }
}

function parseLocal(value: string): Date | null {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return null
  return date
}

function isHourAligned(date: Date): boolean {
  return date.getMinutes() === 0 && date.getSeconds() === 0
}

export function validateHorizon(start: string, end: string): string | null {
  if (!start || !end) return 'Set a start and end time.'
  const startDate = parseLocal(start)
  const endDate = parseLocal(end)
  if (!startDate || !endDate) return 'Enter a valid date and time.'
  if (!isHourAligned(startDate) || !isHourAligned(endDate)) {
    return 'Use on-the-hour times. Predictions are hourly.'
  }
  if (endDate.getTime() <= startDate.getTime()) return 'End must be after start.'
  const minutes = (endDate.getTime() - startDate.getTime()) / 60_000
  if (minutes < MIN_MINUTES) return 'Use a window of at least 1 hour.'
  if (minutes > MAX_MINUTES) return 'Use a window of at most 2 hours.'
  return null
}

export function formatClock(value: string): string {
  return new Intl.DateTimeFormat('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  }).format(new Date(value))
}

export function formatRangeLabel(start: string, end: string): string {
  const date = new Intl.DateTimeFormat('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  }).format(new Date(start))
  return `${date}, ${formatClock(start)}–${formatClock(end)}`
}

export function hourKey(value: string): number {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return Number.NaN
  date.setMinutes(0, 0, 0)
  return date.getTime()
}

export function iterateWindow(start: string, end: string): Date[] {
  const startDate = parseLocal(start)
  const endDate = parseLocal(end)
  if (!startDate || !endDate) return []
  const step = 60 * 60 * 1000
  const points: Date[] = []
  for (let time = startDate.getTime(); time <= endDate.getTime(); time += step) {
    points.push(new Date(time))
  }
  return points
}


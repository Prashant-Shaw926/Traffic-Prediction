const MIN_MINUTES = 15
const MAX_MINUTES = 120

export function toDateTimeLocal(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`
}

export function defaultWindow(): { start: string; end: string } {
  const start = new Date()
  start.setSeconds(0, 0)
  start.setMilliseconds(0)
  start.setHours(7, 0, 0, 0)

  if (start.getTime() <= Date.now()) {
    start.setHours(17, 0, 0, 0)
  }
  if (start.getTime() <= Date.now()) {
    start.setDate(start.getDate() + 1)
    start.setHours(8, 0, 0, 0)
  }

  const end = new Date(start.getTime() + 2 * 60 * 60 * 1000)
  return { start: toDateTimeLocal(start), end: toDateTimeLocal(end) }
}

export function validateHorizon(start: string, end: string): string | null {
  if (!start || !end) return 'Set a start and end time.'
  const startMs = new Date(start).getTime()
  const endMs = new Date(end).getTime()
  if (Number.isNaN(startMs) || Number.isNaN(endMs)) {
    return 'Enter a valid date and time.'
  }
  if (endMs <= startMs) return 'End must be after start.'
  const minutes = (endMs - startMs) / 60_000
  if (minutes < MIN_MINUTES) return 'Use a window of at least 15 minutes.'
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
  }).format(new Date(start))
  return `${date}, ${formatClock(start)}–${formatClock(end)}`
}

export function iterateWindow(start: string, end: string): Date[] {
  const startMs = new Date(start).getTime()
  const endMs = new Date(end).getTime()
  const step = MIN_MINUTES * 60_000
  const points: Date[] = []
  for (let time = startMs; time <= endMs; time += step) {
    points.push(new Date(time))
  }
  const last = points[points.length - 1]
  if (!last || last.getTime() !== endMs) {
    points.push(new Date(endMs))
  }
  return points
}

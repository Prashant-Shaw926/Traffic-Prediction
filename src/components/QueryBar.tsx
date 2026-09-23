import type { Junction } from '../types/traffic'

type QueryBarProps = {
  locations: Junction[]
  selectedLocationId: string | null
  start: string
  end: string
  isLoading: boolean
  error: string | null
  onLocationChange: (id: string) => void
  onStartChange: (value: string) => void
  onEndChange: (value: string) => void
  onPredict: () => void
}

const fieldClass =
  'h-9 w-full border-0 border-b border-line bg-transparent text-sm text-ink outline-none transition-colors focus:border-accent'

export const QueryBar = ({
  locations,
  selectedLocationId,
  start,
  end,
  isLoading,
  error,
  onLocationChange,
  onStartChange,
  onEndChange,
  onPredict,
}: QueryBarProps) => {
  return (
    <form
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault()
        onPredict()
      }}
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-[1.2fr_1fr_1fr_auto] lg:items-end">
        <label className="block min-w-0">
          <span className="mb-1 block text-[11px] uppercase tracking-[0.14em] text-muted">
            Junction
          </span>
          <select
            data-testid="location-select"
            className={fieldClass}
            value={selectedLocationId ?? ''}
            onChange={(event) => onLocationChange(event.target.value)}
          >
            <option value="">Select junction</option>
            {locations.map((junction) => (
              <option key={junction.id} value={junction.id}>
                {junction.name} — {junction.area}
              </option>
            ))}
          </select>
        </label>

        <label className="block min-w-0">
          <span className="mb-1 block text-[11px] uppercase tracking-[0.14em] text-muted">
            Start
          </span>
          <input
            data-testid="time-start-input"
            type="datetime-local"
            className={fieldClass}
            value={start}
            onChange={(event) => onStartChange(event.target.value)}
          />
        </label>

        <label className="block min-w-0">
          <span className="mb-1 block text-[11px] uppercase tracking-[0.14em] text-muted">
            End
          </span>
          <input
            data-testid="time-end-input"
            type="datetime-local"
            className={fieldClass}
            value={end}
            onChange={(event) => onEndChange(event.target.value)}
          />
        </label>

        <button
          data-testid="predict-submit-btn"
          type="submit"
          disabled={isLoading}
          className="h-9 bg-accent px-4 text-sm font-medium text-canvas transition-[filter,opacity] duration-200 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {isLoading ? 'Predicting…' : 'Predict'}
        </button>
      </div>
      {error ? (
        <p className="text-sm text-severe" role="alert">
          {error}
        </p>
      ) : (
        <p className="text-xs text-muted">Window must be 15 minutes to 2 hours.</p>
      )}
    </form>
  )
}

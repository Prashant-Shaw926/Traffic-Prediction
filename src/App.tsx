import { useEffect, useMemo, useState } from 'react'
import { mockTrafficApi } from './api/traffic'
import { AppShell } from './components/AppShell'
import { ForecastChart } from './components/ForecastChart'
import { HistoryCompare } from './components/HistoryCompare'
import { LocationMap } from './components/LocationMap'
import { QueryBar } from './components/QueryBar'
import type {
  CongestionLevel,
  Junction,
  PredictResponse,
  PredictionPoint,
} from './types/traffic'
import { CONGESTION_COLORS, CONGESTION_LABELS } from './types/traffic'
import { defaultWindow, formatRangeLabel, validateHorizon } from './utils/horizon'

const INITIAL_WINDOW = defaultWindow()

type Status = 'idle' | 'loading' | 'ready'

const App = () => {
  const [locations, setLocations] = useState<Junction[]>([])
  const [selectedLocationId, setSelectedLocationId] = useState<string | null>(
    null,
  )
  const [start, setStart] = useState(INITIAL_WINDOW.start)
  const [end, setEnd] = useState(INITIAL_WINDOW.end)
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [history, setHistory] = useState<PredictionPoint[] | null>(null)

  useEffect(() => {
    let cancelled = false

    mockTrafficApi
      .getLocations()
      .then((items) => {
        if (!cancelled) setLocations(items)
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Could not load junctions.')
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  const selected =
    locations.find((item) => item.id === selectedLocationId) ?? null

  const congestionById = useMemo(() => {
    if (!result) return {}
    return { [result.locationId]: result.peakCongestion } satisfies Partial<
      Record<string, CongestionLevel>
    >
  }, [result])

  const handlePredict = () => {
    const validation = validateHorizon(start, end)
    if (!selectedLocationId) {
      setError('Select a junction.')
      return
    }
    if (validation) {
      setError(validation)
      return
    }

    setError(null)
    setStatus('loading')

    void Promise.all([
      mockTrafficApi.predict({ locationId: selectedLocationId, start, end }),
      mockTrafficApi.getHistory(selectedLocationId, start, end),
    ])
      .then(([prediction, historical]) => {
        setResult(prediction)
        setHistory(historical)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        setResult(null)
        setHistory(null)
        setStatus('idle')
        setError(err instanceof Error ? err.message : 'Prediction failed.')
      })
  }

  const showResult =
    status === 'ready' &&
    result !== null &&
    history !== null &&
    result.locationId === selectedLocationId

  return (
    <AppShell
      query={
        <QueryBar
          locations={locations}
          selectedLocationId={selectedLocationId}
          start={start}
          end={end}
          isLoading={status === 'loading'}
          error={error}
          onLocationChange={(id) => {
            setSelectedLocationId(id || null)
            setError(null)
          }}
          onStartChange={(value) => {
            setStart(value)
            setError(null)
          }}
          onEndChange={(value) => {
            setEnd(value)
            setError(null)
          }}
          onPredict={handlePredict}
        />
      }
      map={
        <LocationMap
          locations={locations}
          selectedLocationId={selectedLocationId}
          congestionById={congestionById}
          onSelect={(id) => {
            setSelectedLocationId(id)
            setError(null)
          }}
        />
      }
      inspector={
        <Inspector
          selected={selected}
          start={start}
          end={end}
          status={status}
          showResult={showResult}
          result={result}
          history={history}
        />
      }
    />
  )
}

type InspectorProps = {
  selected: Junction | null
  start: string
  end: string
  status: Status
  showResult: boolean
  result: PredictResponse | null
  history: PredictionPoint[] | null
}

const Inspector = ({
  selected,
  start,
  end,
  status,
  showResult,
  result,
  history,
}: InspectorProps) => {
  if (!selected) {
    return (
      <div className="px-5 py-5">
        <p className="text-sm text-muted">
          Select a junction and time range, then predict.
        </p>
      </div>
    )
  }

  if (status === 'loading') {
    return (
      <div className="px-5 py-5">
        <p className="text-[11px] uppercase tracking-[0.14em] text-muted">
          {selected.name}
        </p>
        <p className="mt-3 text-sm text-muted">Running prediction…</p>
      </div>
    )
  }

  if (!showResult || !result || !history) {
    return (
      <div className="px-5 py-5">
        <p className="text-[11px] uppercase tracking-[0.14em] text-muted">
          {selected.area}
        </p>
        <h2 className="mt-1 text-xl font-medium tracking-tight">{selected.name}</h2>
        <p className="mt-3 text-sm text-muted">
          Set a 15 minute to 2 hour window, then predict.
        </p>
      </div>
    )
  }

  return (
    <div className="px-5 py-5">
      <p className="text-[11px] uppercase tracking-[0.14em] text-muted">
        {selected.area}
      </p>
      <h2 className="mt-1 text-xl font-medium tracking-tight">{selected.name}</h2>
      <p className="mt-1 text-sm text-muted">{formatRangeLabel(start, end)}</p>

      <div className="mt-6 flex items-end justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.14em] text-muted">
            Peak congestion
          </p>
          <p
            className="mt-1 text-lg font-medium"
            style={{ color: CONGESTION_COLORS[result.peakCongestion] }}
          >
            {CONGESTION_LABELS[result.peakCongestion]}
          </p>
        </div>
        <div className="text-right">
          <p className="text-[11px] uppercase tracking-[0.14em] text-muted">
            Peak volume
          </p>
          <p className="mt-1 font-mono text-lg tabular-nums">
            {result.peakVolume}
            <span className="ml-1 text-xs text-muted">veh</span>
          </p>
        </div>
      </div>

      <section className="mt-8">
        <h3 className="text-[11px] uppercase tracking-[0.14em] text-muted">
          Predicted volume
        </h3>
        <div className="mt-3">
          <ForecastChart points={result.points} />
        </div>
      </section>

      <section className="mt-8">
        <h3 className="text-[11px] uppercase tracking-[0.14em] text-muted">
          Actual vs predicted
        </h3>
        <div className="mt-3">
          <HistoryCompare predicted={result.points} history={history} />
        </div>
      </section>
    </div>
  )
}

export default App

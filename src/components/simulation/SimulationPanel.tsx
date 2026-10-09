import { useRef, useState } from 'react'
import { runSimulation } from '../../api/simulation'
import type { SimulationResponse } from '../../types/simulation'
import { AppShell } from '../AppShell'
import { ViewSwitch, type WorkspaceView } from '../ViewSwitch'
import { relationshipKey } from './relationship'
import { TrafficNetwork } from './TrafficNetwork'

const DEFAULT_TIMESTAMP = '2017-06-15T10:00'
const fieldClass =
  'h-9 w-full border-0 border-b border-line bg-transparent text-sm text-ink outline-none transition-colors focus:border-accent'

type SimulationPanelProps = {
  view: WorkspaceView
  onViewChange: (view: WorkspaceView) => void
}

function formatSigned(value: number): string {
  return value.toFixed(3)
}

function formatCount(value: number | null): string {
  if (value === null) return 'Historical value unavailable'
  return value.toFixed(1)
}

export const SimulationPanel = ({ view, onViewChange }: SimulationPanelProps) => {
  const [datetime, setDatetime] = useState(DEFAULT_TIMESTAMP)
  const [rawMode, setRawMode] = useState(false)
  const [rawTimestamp, setRawTimestamp] = useState('')
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SimulationResponse | null>(null)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const inFlight = useRef(false)

  const handleRun = () => {
    if (inFlight.current) return
    if (!rawMode && !datetime) {
      setError('Set a historical timestamp.')
      return
    }
    inFlight.current = true
    setError(null)
    setResult(null)
    setSelectedKey(null)
    setStatus('loading')
    const request = rawMode
      ? runSimulation(rawTimestamp, { exact: true })
      : runSimulation(datetime)
    void request
      .then((response) => {
        setResult(response)
        setSelectedKey(response.relationships[0] ? relationshipKey(response.relationships[0]) : null)
        setStatus('ready')
      })
      .catch((err: unknown) => {
        setResult(null)
        setStatus('idle')
        setError(err instanceof Error ? err.message : 'Simulation failed.')
      })
      .finally(() => {
        inFlight.current = false
      })
  }

  const selected =
    result?.relationships.find((item) => relationshipKey(item) === selectedKey) ?? null

  return (
    <AppShell
      nav={<ViewSwitch view={view} onViewChange={onViewChange} />}
      query={
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault()
            handleRun()
          }}
        >
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
            <label className="block min-w-0">
              <span className="mb-1 block text-[11px] uppercase tracking-[0.14em] text-muted">
                Historical timestamp
              </span>
              <input
                data-testid="simulation-time-input"
                type="datetime-local"
                step={3600}
                className={fieldClass}
                value={datetime}
                onChange={(event) => {
                  setDatetime(event.target.value)
                  setError(null)
                }}
              />
            </label>
            <button
              data-testid="simulation-submit-btn"
              type="submit"
              disabled={status === 'loading'}
              className="h-9 bg-accent px-4 text-sm font-medium text-canvas transition-[filter,opacity] duration-200 hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {status === 'loading' ? 'Running…' : 'Run Simulation'}
            </button>
          </div>
          <label className="flex items-center gap-2 text-xs text-muted">
            <input
              data-testid="simulation-raw-mode"
              type="checkbox"
              checked={rawMode}
              onChange={(event) => {
                setRawMode(event.target.checked)
                setError(null)
              }}
            />
            Submit a raw timestamp
          </label>
          {rawMode ? (
            <label className="block min-w-0">
              <span className="mb-1 block text-[11px] uppercase tracking-[0.14em] text-muted">
                Raw timestamp
              </span>
              <input
                data-testid="simulation-raw-time-input"
                type="text"
                autoComplete="off"
                spellCheck={false}
                className={fieldClass}
                value={rawTimestamp}
                onChange={(event) => {
                  setRawTimestamp(event.target.value)
                  setError(null)
                }}
              />
            </label>
          ) : null}
          {error ? (
            <p className="text-sm text-severe" role="alert" data-testid="simulation-error">
              {error}
            </p>
          ) : (
            <p className="text-xs text-muted">
              Historical Traffic Network Simulation. The selected timestamp drives
              the Python simulation. This is not a live traffic feed. Dataset ends
              30 Jun 2017.
              {rawMode
                ? ' The raw timestamp is sent exactly as typed.'
                : ''}
            </p>
          )}
        </form>
      }
      map={
        result ? (
          <TrafficNetwork
            junctions={result.junctions}
            relationships={result.relationships}
            selectedKey={selectedKey}
            onSelect={setSelectedKey}
          />
        ) : (
          <div className="flex h-full items-center justify-center px-6 text-sm text-muted">
            {status === 'loading'
              ? 'Running historical simulation…'
              : 'Choose a timestamp, then run the simulation.'}
          </div>
        )
      }
      inspector={
        <div className="px-5 py-5">
          <h2 className="text-lg font-medium tracking-tight">
            Historical Traffic Network Simulation
          </h2>
          <p className="mt-2 text-xs leading-5 text-muted">
            Relationships are historical lagged statistical associations. Correlation
            does not establish a physical road connection or causation. The influence
            index is a mathematical index, not a predicted vehicle count.
          </p>
          {!result ? (
            <p className="mt-6 text-sm text-muted">No simulation result yet.</p>
          ) : (
            <>
              <p className="mt-4 text-xs text-muted">
                {result.metadata.model.replaceAll('_', ' ')} · {result.metadata.features}{' '}
                features · lookback {result.metadata.lookback} hours
                {result.metadata.historical_simulation ? ' · historical simulation' : ''}
              </p>
              <p className="mt-1 text-xs text-muted">
                Window {result.metadata.window_start.replace('T', ' ')} to{' '}
                {result.metadata.window_end.replace('T', ' ')}
              </p>
              <section className="mt-6">
                <h3 className="text-[11px] uppercase tracking-[0.14em] text-muted">
                  Junctions
                </h3>
                <ul className="mt-3 space-y-3">
                  {result.junctions.map((junction) => (
                    <li
                      key={junction.junction_id}
                      data-testid={`junction-card-${junction.junction_id}`}
                      className="border border-line px-3 py-3"
                    >
                      <p className="text-sm font-medium">Junction {junction.junction_id}</p>
                      <p className="mt-2 text-xs text-muted">
                        Predicted traffic{' '}
                        <span className="font-mono text-ink">
                          {junction.predicted_traffic.toFixed(2)} veh
                        </span>
                      </p>
                      <p className="mt-1 text-xs text-muted">
                        Historical traffic{' '}
                        <span className="font-mono text-ink">
                          {junction.historical_traffic === null
                            ? 'Historical value unavailable'
                            : `${formatCount(junction.historical_traffic)} veh`}
                        </span>
                      </p>
                      <p className="mt-1 text-xs text-muted">
                        Influence index{' '}
                        <span className="font-mono text-ink">
                          {junction.incoming_influence === null
                            ? 'Unavailable'
                            : junction.incoming_influence.toFixed(2)}
                        </span>
                      </p>
                      <p className="mt-1 text-xs text-muted">
                        Valid sources {junction.valid_source_count}
                      </p>
                    </li>
                  ))}
                </ul>
              </section>
              <section className="mt-6">
                <h3 className="text-[11px] uppercase tracking-[0.14em] text-muted">
                  Selected relationship
                </h3>
                {selected ? (
                  <div className="mt-3 text-xs leading-5 text-muted" data-testid="selected-relationship">
                    <p className="text-sm text-ink">
                      {selected.source_junction} → {selected.target_junction}
                    </p>
                    <p>Signed correlation {formatSigned(selected.correlation)}</p>
                    <p>Absolute correlation {formatSigned(selected.absolute_correlation)}</p>
                    <p>Lag {selected.lag_hours} hours</p>
                    <p>Paired observations {selected.paired_observations}</p>
                  </div>
                ) : (
                  <p className="mt-3 text-sm text-muted">
                    {result.relationships.length === 0
                      ? 'No relationships were returned.'
                      : 'Select a relationship.'}
                  </p>
                )}
                <ul className="mt-3 max-h-40 space-y-1 overflow-y-auto">
                  {result.relationships.map((item) => {
                    const key = relationshipKey(item)
                    return (
                      <li key={key}>
                        <button
                          type="button"
                          data-testid={`relationship-row-${key}`}
                          className={`w-full px-2 py-1 text-left text-xs ${
                            key === selectedKey ? 'bg-panel text-ink' : 'text-muted'
                          }`}
                          onClick={() => setSelectedKey(key)}
                        >
                          {item.source_junction} → {item.target_junction}:{' '}
                          {formatSigned(item.correlation)} · lag {item.lag_hours}h
                        </button>
                      </li>
                    )
                  })}
                </ul>
              </section>
              <section className="mt-6">
                <h3 className="text-[11px] uppercase tracking-[0.14em] text-muted">
                  Influence index
                </h3>
                <p className="mt-2 text-xs text-muted">
                  Score kind: {result.influence_result.metadata.score_kind}
                </p>
                <ul className="mt-3 space-y-3">
                  {result.influence_result.junction_influences.map((item) => (
                    <li key={item.junction_id} className="text-xs text-muted">
                      <p className="text-ink">
                        Junction {item.junction_id}:{' '}
                        {item.incoming_influence === null
                          ? 'Unavailable'
                          : item.incoming_influence.toFixed(2)}
                      </p>
                      {item.contributions.map((contribution) => (
                        <p
                          key={`${contribution.source_junction}-${contribution.lag_hours}`}
                          className="mt-1"
                        >
                          Source {contribution.source_junction}: traffic{' '}
                          {contribution.source_traffic.toFixed(1)}, signed{' '}
                          {formatSigned(contribution.signed_correlation)}, lag{' '}
                          {contribution.lag_hours}h, weight{' '}
                          {formatSigned(contribution.relationship_weight)}, contribution{' '}
                          {contribution.contribution.toFixed(2)}
                        </p>
                      ))}
                    </li>
                  ))}
                </ul>
                {result.influence_result.excluded_contributions.length > 0 ? (
                  <div className="mt-3">
                    <p className="text-xs text-ink">Excluded contributions</p>
                    {result.influence_result.excluded_contributions.map((item) => (
                      <p
                        key={`${item.source_junction}-${item.target_junction}-${item.lag_hours}`}
                        className="mt-1 text-xs text-muted"
                      >
                        {item.source_junction} → {item.target_junction}: {item.reason}
                      </p>
                    ))}
                  </div>
                ) : null}
              </section>
            </>
          )}
        </div>
      }
    />
  )
}

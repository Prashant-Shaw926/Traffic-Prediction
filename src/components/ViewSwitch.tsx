type WorkspaceView = 'prediction' | 'simulation'

type ViewSwitchProps = {
  view: WorkspaceView
  onViewChange: (view: WorkspaceView) => void
}

const buttonClass = (active: boolean) =>
  `h-7 px-2 text-[11px] uppercase tracking-[0.12em] ${
    active ? 'bg-accent text-canvas' : 'text-muted hover:text-ink'
  }`

export const ViewSwitch = ({ view, onViewChange }: ViewSwitchProps) => {
  return (
    <div className="flex border border-line" role="group" aria-label="Workspace">
      <button
        type="button"
        data-testid="view-prediction-btn"
        className={buttonClass(view === 'prediction')}
        aria-pressed={view === 'prediction'}
        onClick={() => onViewChange('prediction')}
      >
        Prediction
      </button>
      <button
        type="button"
        data-testid="view-simulation-btn"
        className={buttonClass(view === 'simulation')}
        aria-pressed={view === 'simulation'}
        onClick={() => onViewChange('simulation')}
      >
        Historical Traffic Network Simulation
      </button>
    </div>
  )
}

export type { WorkspaceView }

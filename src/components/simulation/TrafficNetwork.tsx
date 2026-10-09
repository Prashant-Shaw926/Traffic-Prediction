import type { SimulationJunction, SimulationRelationship } from '../../types/simulation'
import { relationshipKey } from './relationship'

type TrafficNetworkProps = {
  junctions: SimulationJunction[]
  relationships: SimulationRelationship[]
  selectedKey: string | null
  onSelect: (key: string) => void
}

type Point = { x: number; y: number }

const WIDTH = 640
const HEIGHT = 420

function nodePositions(junctions: SimulationJunction[]): Map<number, Point> {
  const ordered = [...junctions].sort((left, right) => left.junction_id - right.junction_id)
  const positions = new Map<number, Point>()
  const cx = WIDTH / 2
  const cy = HEIGHT / 2
  const radius = Math.min(WIDTH, HEIGHT) * 0.32
  ordered.forEach((junction, index) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / Math.max(ordered.length, 1)
    positions.set(junction.junction_id, {
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
    })
  })
  return positions
}

function edgeStyle(strength: number, selected: boolean): { width: number; opacity: number } {
  const clamped = Math.min(Math.max(strength, 0), 1)
  return {
    width: selected ? 2 + clamped * 6 : 1 + clamped * 5,
    opacity: 0.35 + clamped * 0.6,
  }
}

export const TrafficNetwork = ({
  junctions,
  relationships,
  selectedKey,
  onSelect,
}: TrafficNetworkProps) => {
  const positions = nodePositions(junctions)

  return (
    <div className="flex h-full min-h-[50svh] flex-col bg-canvas" data-testid="simulation-network">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="h-full min-h-0 w-full flex-1"
        role="img"
        aria-label="Directed junction relationships from the simulation response"
      >
        <defs>
          <marker
            id="relationship-arrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#e6e8eb" />
          </marker>
        </defs>
        {relationships.map((item) => {
          const source = positions.get(item.source_junction)
          const target = positions.get(item.target_junction)
          if (!source || !target) return null
          const key = relationshipKey(item)
          const selected = key === selectedKey
          const style = edgeStyle(item.absolute_correlation, selected)
          const dx = target.x - source.x
          const dy = target.y - source.y
          const length = Math.hypot(dx, dy) || 1
          const trim = 28
          const x2 = target.x - (dx / length) * trim
          const y2 = target.y - (dy / length) * trim
          return (
            <line
              key={key}
              data-testid={`relationship-edge-${key}`}
              x1={source.x}
              y1={source.y}
              x2={x2}
              y2={y2}
              stroke={selected ? '#f5a524' : '#e6e8eb'}
              strokeWidth={style.width}
              strokeOpacity={style.opacity}
              markerEnd="url(#relationship-arrow)"
              role="button"
              tabIndex={0}
              aria-label={`Junction ${item.source_junction} to ${item.target_junction}, correlation ${item.correlation.toFixed(3)}, lag ${item.lag_hours} hours`}
              onClick={() => onSelect(key)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onSelect(key)
                }
              }}
            />
          )
        })}
        {junctions.map((junction) => {
          const point = positions.get(junction.junction_id)
          if (!point) return null
          return (
            <g key={junction.junction_id} data-testid={`network-node-${junction.junction_id}`}>
              <circle cx={point.x} cy={point.y} r="22" fill="#111318" stroke="#f5a524" strokeWidth="2" />
              <text
                x={point.x}
                y={point.y + 4}
                textAnchor="middle"
                fill="#e6e8eb"
                fontSize="13"
              >
                {junction.junction_id}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="flex flex-wrap items-center gap-4 border-t border-line px-4 py-3 text-xs text-muted">
        <span className="inline-flex items-center gap-2">
          <svg width="42" height="10" aria-hidden="true">
            <line x1="0" y1="5" x2="32" y2="5" stroke="#e6e8eb" strokeWidth="4" />
            <path d="M 32 1 L 42 5 L 32 9 z" fill="#e6e8eb" />
          </svg>
          Arrow points from source to target
        </span>
        <span>Line thickness shows absolute correlation, not direction</span>
        <span>Layout is a display arrangement, not a road map</span>
      </div>
    </div>
  )
}

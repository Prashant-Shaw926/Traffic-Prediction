import {
  CONGESTION_COLORS,
  CONGESTION_LABELS,
  CONGESTION_ORDER,
} from '../types/traffic'
import type { CongestionLevel } from '../types/traffic'

export const CongestionLegend = () => {
  return (
    <ul className="pointer-events-none absolute bottom-8 left-3 z-[400] flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted">
      {CONGESTION_ORDER.map((level: CongestionLevel) => (
        <li key={level} className="flex items-center gap-1.5">
          <span
            className="inline-block size-2 rounded-full"
            style={{ backgroundColor: CONGESTION_COLORS[level] }}
          />
          {CONGESTION_LABELS[level]}
        </li>
      ))}
    </ul>
  )
}

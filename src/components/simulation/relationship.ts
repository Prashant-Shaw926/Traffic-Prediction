import type { SimulationRelationship } from '../../types/simulation'

export function relationshipKey(item: SimulationRelationship): string {
  return `${item.source_junction}-${item.target_junction}-${item.lag_hours}`
}

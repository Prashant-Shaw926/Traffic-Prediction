import { JUNCTIONS, buildSeries, peakOf } from '../data/mockTraffic'
import type { PredictRequest, PredictResponse, TrafficApi } from '../types/traffic'
import { validateHorizon } from '../utils/horizon'

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

function mockLatency(): Promise<void> {
  return wait(300 + Math.floor(Math.random() * 300))
}

function requireJunction(locationId: string) {
  const junction = JUNCTIONS.find((item) => item.id === locationId)
  if (!junction) {
    throw new Error('Unknown junction.')
  }
  return junction
}

export const mockTrafficApi: TrafficApi = {
  async getLocations() {
    await mockLatency()
    return JUNCTIONS
  },

  async predict(request: PredictRequest): Promise<PredictResponse> {
    await mockLatency()
    const error = validateHorizon(request.start, request.end)
    if (error) throw new Error(error)
    requireJunction(request.locationId)

    const points = buildSeries(
      request.locationId,
      request.start,
      request.end,
      false,
    )
    const peak = peakOf(points)

    return {
      locationId: request.locationId,
      start: request.start,
      end: request.end,
      points,
      peakCongestion: peak.peakCongestion,
      peakVolume: peak.peakVolume,
    }
  },

  async getHistory(locationId, start, end) {
    await mockLatency()
    const error = validateHorizon(start, end)
    if (error) throw new Error(error)
    requireJunction(locationId)
    return buildSeries(locationId, start, end, true)
  },
}

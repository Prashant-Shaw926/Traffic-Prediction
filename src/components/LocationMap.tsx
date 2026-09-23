import { useEffect } from 'react'
import {
  CircleMarker,
  MapContainer,
  TileLayer,
  Tooltip,
  useMap,
} from 'react-leaflet'
import {
  CONGESTION_COLORS,
  type CongestionLevel,
  type Junction,
} from '../types/traffic'
import { CongestionLegend } from './CongestionLegend'

const MAP_CENTER: [number, number] = [12.9224, 77.6354]
const IDLE_FILL = '#5c6270'

type LocationMapProps = {
  locations: Junction[]
  selectedLocationId: string | null
  congestionById: Partial<Record<string, CongestionLevel>>
  onSelect: (locationId: string) => void
}

const FocusJunction = ({ junction }: { junction: Junction | null }) => {
  const map = useMap()

  useEffect(() => {
    if (!junction) return
    map.flyTo([junction.lat, junction.lng], Math.max(map.getZoom(), 14), {
      duration: 0.55,
    })
  }, [junction, map])

  return null
}

const InvalidateSize = () => {
  const map = useMap()

  useEffect(() => {
    const id = window.setTimeout(() => map.invalidateSize(), 80)
    return () => window.clearTimeout(id)
  }, [map])

  return null
}

export const LocationMap = ({
  locations,
  selectedLocationId,
  congestionById,
  onSelect,
}: LocationMapProps) => {
  const selected = locations.find((item) => item.id === selectedLocationId) ?? null

  return (
    <div data-testid="location-map" className="absolute inset-0">
      <MapContainer
        center={MAP_CENTER}
        zoom={13}
        scrollWheelZoom
        className="h-full w-full"
      >
        <TileLayer
          attribution="Tiles &copy; Esri"
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        />
        <InvalidateSize />
        <FocusJunction junction={selected} />
        {locations.map((junction) => {
          const selectedDot = junction.id === selectedLocationId
          const congestion = congestionById[junction.id]
          const fill = congestion ? CONGESTION_COLORS[congestion] : IDLE_FILL
          return (
            <CircleMarker
              key={junction.id}
              center={[junction.lat, junction.lng]}
              radius={selectedDot ? 11 : 7}
              className="junction-dot"
              pathOptions={{
                color: selectedDot ? '#e6e8eb' : fill,
                fillColor: fill,
                fillOpacity: 0.92,
                weight: selectedDot ? 3 : 2,
              }}
              eventHandlers={{
                click: () => onSelect(junction.id),
              }}
            >
              <Tooltip
                direction="top"
                offset={[0, -6]}
                className="junction-tip"
              >
                {`${junction.name} · ${junction.area}`}
              </Tooltip>
            </CircleMarker>
          )
        })}
      </MapContainer>
      <CongestionLegend />
    </div>
  )
}

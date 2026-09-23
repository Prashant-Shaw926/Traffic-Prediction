import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { PredictionPoint } from '../types/traffic'
import { formatClock } from '../utils/horizon'

type ForecastChartProps = {
  points: PredictionPoint[]
}

export const ForecastChart = ({ points }: ForecastChartProps) => {
  const data = points.map((point) => ({
    label: formatClock(point.timestamp),
    predicted: point.predicted,
  }))

  return (
    <div data-testid="forecast-chart" className="h-44 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="predFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f5a524" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#f5a524" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#23262e" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: '#8b909a', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#8b909a', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={36}
          />
          <Tooltip
            cursor={{ stroke: '#23262e' }}
            contentStyle={{
              background: '#111318',
              border: '1px solid #23262e',
              borderRadius: 0,
              fontSize: 12,
            }}
            labelStyle={{ color: '#8b909a' }}
            formatter={(value) => [`${value ?? 0} veh`, 'Predicted']}
          />
          <Area
            type="monotone"
            dataKey="predicted"
            stroke="#f5a524"
            fill="url(#predFill)"
            strokeWidth={2.5}
            animationDuration={800}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

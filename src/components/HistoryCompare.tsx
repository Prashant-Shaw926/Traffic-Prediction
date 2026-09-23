import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { PredictionPoint } from '../types/traffic'
import { formatClock } from '../utils/horizon'

type HistoryCompareProps = {
  predicted: PredictionPoint[]
  history: PredictionPoint[]
}

export const HistoryCompare = ({ predicted, history }: HistoryCompareProps) => {
  const data = predicted.map((point, index) => ({
    label: formatClock(point.timestamp),
    predicted: point.predicted,
    actual: history[index]?.actual ?? history[index]?.predicted ?? 0,
  }))

  return (
    <div data-testid="history-compare-chart" className="h-44 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
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
            formatter={(value, name) => [
              `${value ?? 0} veh`,
              name === 'actual' ? 'Actual' : 'Predicted',
            ]}
          />
          <Legend
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, color: '#8b909a' }}
            formatter={(value) => (value === 'actual' ? 'Actual' : 'Predicted')}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#f5a524"
            strokeWidth={2}
            dot={false}
            animationDuration={800}
          />
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#8b909a"
            strokeWidth={2}
            dot={false}
            animationDuration={800}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface Series {
  key: string;
  name: string;
  color: string;
  type?: 'line' | 'area';
}

interface LineChartAutoRefreshProps {
  data: Array<Record<string, unknown>>;
  series: Series[];
  height?: number;
  xKey?: string;
}

export function LineChartAutoRefresh({ data, series, height = 280, xKey = 'generation' }: LineChartAutoRefreshProps) {
  const areaMode = series.some((item) => item.type === 'area');
  const Chart = areaMode ? AreaChart : LineChart;
  return (
    <div className="h-full w-full" style={{ minHeight: height }}>
      <ResponsiveContainer width="100%" height={height}>
        <Chart data={data} margin={{ top: 10, right: 18, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey={xKey} tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} width={48} />
          <Tooltip />
          <Legend />
          {series.map((item) =>
            item.type === 'area' ? (
              <Area
                key={item.key}
                type="monotone"
                dataKey={item.key}
                name={item.name}
                stackId="diversity"
                stroke={item.color}
                fill={item.color}
                fillOpacity={0.28}
              />
            ) : (
              <Line
                key={item.key}
                type="monotone"
                dataKey={item.key}
                name={item.name}
                dot={false}
                stroke={item.color}
                strokeWidth={2}
              />
            ),
          )}
        </Chart>
      </ResponsiveContainer>
    </div>
  );
}

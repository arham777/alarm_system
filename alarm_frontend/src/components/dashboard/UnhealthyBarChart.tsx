import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { UnhealthyBar } from '@/types/dashboard';

interface UnhealthyBarChartProps {
  data: UnhealthyBar[];
  threshold: number;
  topN: 1 | 3;
  onTopNChange: (value: 1 | 3) => void;
  isLoading?: boolean;
}

export function UnhealthyBarChart({ 
  data, 
  threshold, 
  topN, 
  onTopNChange, 
  isLoading = false 
}: UnhealthyBarChartProps) {
  const formatTooltip = (value: number, name: string, props: any) => {
    const { payload } = props;
    if (!payload) return null;

    return [
      <div key="tooltip" className="bg-dashboard-chart-tooltip-bg p-3 rounded shadow-lg border">
        <p className="font-medium text-foreground">{payload.source}</p>
        <p className="text-sm text-muted-foreground">
          Hits: <span className="font-medium text-foreground">{payload.hits}</span>
        </p>
        <p className="text-sm text-muted-foreground">
          Over threshold by: <span className="font-medium text-red-600">{payload.over_by}</span>
        </p>
        <p className="text-xs text-muted-foreground mt-1">
          {new Date(payload.bin_start).toLocaleString()} - {new Date(payload.bin_end).toLocaleString()}
        </p>
      </div>
    ];
  };

  if (isLoading) {
    return (
      <Card className="shadow-metric-card">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <div className="h-6 w-48 bg-muted animate-pulse rounded mb-2" />
              <div className="h-4 w-64 bg-muted animate-pulse rounded" />
            </div>
            <div className="flex gap-2">
              <div className="h-8 w-12 bg-muted animate-pulse rounded" />
              <div className="h-8 w-12 bg-muted animate-pulse rounded" />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80 bg-muted animate-pulse rounded" />
        </CardContent>
      </Card>
    );
  }

  const isEmpty = data.length === 0;

  return (
    <Card className="shadow-metric-card bg-dashboard-metric-card-bg">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg font-semibold text-foreground">
              Unhealthy Sources
            </CardTitle>
            <CardDescription>
              Sources exceeding threshold of {threshold} hits
            </CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant={topN === 1 ? 'default' : 'outline'}
              size="sm"
              onClick={() => onTopNChange(1)}
            >
              Top 1
            </Button>
            <Button
              variant={topN === 3 ? 'default' : 'outline'}
              size="sm"
              onClick={() => onTopNChange(3)}
            >
              Top 3
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isEmpty ? (
          <div className="h-80 flex items-center justify-center">
            <div className="text-center">
              <p className="text-lg font-medium text-muted-foreground">
                All sources are healthy in the selected window.
              </p>
              <p className="text-sm text-muted-foreground mt-1">
                No sources exceed the threshold of {threshold} hits.
              </p>
            </div>
          </div>
        ) : (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data}
                margin={{
                  top: 20,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid 
                  strokeDasharray="3 3" 
                  stroke="var(--chart-grid)"
                  opacity={0.3}
                />
                <XAxis 
                  dataKey="source" 
                  tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                  axisLine={{ stroke: 'var(--border)' }}
                />
                <YAxis 
                  tick={{ fill: 'var(--muted-foreground)', fontSize: 12 }}
                  axisLine={{ stroke: 'var(--border)' }}
                />
                <Tooltip 
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return formatTooltip(payload[0].value as number, payload[0].name as string, payload[0]);
                    }
                    return null;
                  }}
                  cursor={{ fill: 'var(--accent)', opacity: 0.1 }}
                />
                <ReferenceLine 
                  y={threshold} 
                  stroke="var(--muted-foreground)" 
                  strokeDasharray="5 5"
                  label={{ 
                    value: `Threshold (${threshold})`, 
                    position: 'right',
                    fill: 'var(--muted-foreground)',
                    fontSize: 12
                  }}
                />
                <Bar 
                  dataKey="hits" 
                  fill="var(--primary)"
                  radius={[4, 4, 0, 0]}
                  opacity={0.8}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
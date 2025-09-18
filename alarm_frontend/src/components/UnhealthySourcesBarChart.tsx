import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertTriangle, Clock, RefreshCw, TrendingUp } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts';

interface UnhealthyRecord {
  event_time: string;
  bin_end: string;
  source: string;
  hits: number;
  threshold: number;
  over_by: number;
  rate_per_min: number;
  location_tag?: string;
  condition?: string;
  action?: string;
  priority?: string;
  description?: string;
  value?: number;
  units?: string;
}

interface UnhealthySourcesData {
  count: number;
  records: UnhealthyRecord[];
  isHistoricalData?: boolean;
  note?: string;
}

interface UnhealthySourcesBarChartProps {
  className?: string;
}

const UnhealthySourcesBarChart: React.FC<UnhealthySourcesBarChartProps> = ({ className }) => {
  const [data, setData] = useState<UnhealthySourcesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [sortBy, setSortBy] = useState<'hits' | 'alphabetical'>('hits');
  const [topLimit, setTopLimit] = useState<number>(20);

  useEffect(() => {
    fetchUnhealthySources();
  }, [timeRange]);

  const fetchUnhealthySources = async (skipTimeFilter = false) => {
    try {
      setLoading(true);
      setError(null);
      
      let startTimeStr, endTimeStr;
      
      if (!skipTimeFilter) {
        // Calculate time range
        const endTime = new Date();
        const startTime = new Date();
        
        switch (timeRange) {
          case '1h':
            startTime.setHours(endTime.getHours() - 1);
            break;
          case '6h':
            startTime.setHours(endTime.getHours() - 6);
            break;
          case '24h':
            startTime.setDate(endTime.getDate() - 1);
            break;
          case '7d':
            startTime.setDate(endTime.getDate() - 7);
            break;
          default:
            startTime.setDate(endTime.getDate() - 1);
        }

        startTimeStr = startTime.toISOString();
        endTimeStr = endTime.toISOString();
        console.log(`Fetching unhealthy sources from ${startTimeStr} to ${endTimeStr}`);
      } else {
        console.log('Fetching all historical unhealthy sources (no time filter)');
      }
      
      const { fetchUnhealthySources } = await import('../api/plantHealth');
      const result = await fetchUnhealthySources(
        startTimeStr,
        endTimeStr
      );
      
      console.log('Unhealthy sources API response:', result);
      setData(result);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch unhealthy sources data';
      setError(errorMessage);
      console.error('Error fetching unhealthy sources:', err);
    } finally {
      setLoading(false);
    }
  };

  // Process data for bar chart - Extract actual alarm sources from records
  const processedData = React.useMemo(() => {
    if (!data || !data.records) return [];

    console.log('Processing unhealthy data:', data.records);

    // Group by actual alarm source (not filename)
    const sourceMap = new Map<string, {
      source: string;
      totalHits: number;
      incidents: number;
      maxHits: number;
      avgHits: number;
      latestRecord: UnhealthyRecord;
      priority: string;
      allRecords: UnhealthyRecord[];
    }>();

    data.records.forEach(record => {
      // The API now returns proper alarm source names like EVENT_SCM1B, OP_NASH1, etc.
      let actualSource = record.source;
      
      // Clean up source name only if it still contains file extensions (fallback data)
      if (actualSource.includes('.csv')) {
        actualSource = actualSource.replace('.csv', '');
        console.log('Cleaned filename source:', actualSource);
      }
      
      // If source looks like a filename, try to extract meaningful source name
      if (actualSource.includes('/') || actualSource.includes('\\')) {
        const parts = actualSource.split(/[/\\]/);
        actualSource = parts[parts.length - 1];
      }
      
      // Log the actual source for debugging
      console.log('Processing source:', actualSource, 'with hits:', record.hits);

      const existing = sourceMap.get(actualSource);
      if (existing) {
        existing.totalHits += record.hits;
        existing.incidents += 1;
        existing.maxHits = Math.max(existing.maxHits, record.hits);
        existing.allRecords.push(record);
        // Keep the latest record for details
        if (new Date(record.event_time) > new Date(existing.latestRecord.event_time)) {
          existing.latestRecord = record;
        }
      } else {
        sourceMap.set(actualSource, {
          source: actualSource,
          totalHits: record.hits,
          incidents: 1,
          maxHits: record.hits,
          avgHits: record.hits,
          latestRecord: record,
          priority: record.priority || 'Medium',
          allRecords: [record]
        });
      }
    });

    // Convert to array and calculate averages
    let result = Array.from(sourceMap.values()).map(item => ({
      ...item,
      avgHits: Math.round((item.totalHits / item.incidents) * 10) / 10
    }));

    // Sort data
    if (sortBy === 'hits') {
      result.sort((a, b) => b.totalHits - a.totalHits);
    } else {
      result.sort((a, b) => a.source.localeCompare(b.source));
    }

    // Limit to top N sources to avoid congestion
    result = result.slice(0, topLimit);

    console.log('Processed source data:', result);
    return result;
  }, [data, sortBy, topLimit]);

  // Color mapping for priorities
  const getPriorityColor = (priority: string, hits: number) => {
    switch (priority?.toLowerCase()) {
      case 'high': return '#ef4444'; // Red
      case 'medium': return '#f59e0b'; // Orange
      case 'low': return '#eab308'; // Yellow
      default: 
        // Fallback based on hits if priority not available
        if (hits > 25) return '#ef4444';
        if (hits > 15) return '#f59e0b';
        return '#eab308';
    }
  };

  // Custom tooltip with enhanced source information
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const record = data.latestRecord;
      
      return (
        <div className="bg-white p-4 border rounded-lg shadow-lg max-w-sm border-gray-200">
          <div className="font-semibold text-gray-900 mb-3 text-base">
            🚨 {data.source}
          </div>
          
          <div className="space-y-2 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <div><span className="font-medium text-gray-600">Total Hits:</span></div>
              <div className="font-semibold text-red-600">{data.totalHits}</div>
              
              <div><span className="font-medium text-gray-600">Incidents:</span></div>
              <div>{data.incidents}</div>
              
              <div><span className="font-medium text-gray-600">Max Hits:</span></div>
              <div className="font-semibold">{data.maxHits}</div>
              
              <div><span className="font-medium text-gray-600">Avg Hits:</span></div>
              <div>{data.avgHits}</div>
            </div>
            
            <hr className="my-2" />
            
            <div className="bg-gray-50 p-2 rounded text-xs">
              <div className="font-medium text-gray-700 mb-1">Latest Incident:</div>
              <div><span className="font-medium">Start:</span> {new Date(record.event_time).toLocaleString()}</div>
              <div><span className="font-medium">End:</span> {new Date(record.bin_end).toLocaleString()}</div>
              <div><span className="font-medium">Duration:</span> 10 minutes</div>
            </div>
            
            <div><span className="font-medium text-gray-600">Priority:</span> 
              <span className={`ml-1 px-2 py-1 rounded text-xs font-medium ${
                record.priority === 'High' ? 'bg-red-100 text-red-800' :
                record.priority === 'Medium' ? 'bg-orange-100 text-orange-800' :
                'bg-yellow-100 text-yellow-800'
              }`}>
                {record.priority || 'Medium'}
              </span>
            </div>
            
            {record.location_tag && record.location_tag !== 'Production Area' && (
              <div><span className="font-medium text-gray-600">Location:</span> {record.location_tag}</div>
            )}
            
            {record.condition && record.condition !== 'Alarm Threshold Exceeded' && (
              <div><span className="font-medium text-gray-600">Condition:</span> {record.condition}</div>
            )}
            
            {record.description && record.description !== 'Not Provided' && !record.description.includes('Source exceeded') && (
              <div><span className="font-medium text-gray-600">Description:</span> 
                <div className="text-xs text-gray-600 mt-1">{record.description}</div>
              </div>
            )}
            
            <div className="text-xs text-gray-500 mt-2 bg-red-50 p-2 rounded">
              ⚠️ Threshold: {record.threshold} alarms/10min • Over by: {record.over_by} hits
            </div>
            
            {data.allRecords && data.allRecords.length > 1 && (
              <div className="text-xs text-blue-600 mt-1">
                📊 Total {data.allRecords.length} incidents in selected period
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <RefreshCw className="h-8 w-8 animate-spin text-blue-600 mx-auto mb-2" />
            <p>Loading unhealthy sources...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center text-red-600">
            <AlertTriangle className="h-8 w-8 mx-auto mb-2" />
            <p className="mb-2">{error}</p>
            <Button onClick={fetchUnhealthySources} variant="outline" size="sm">
              <RefreshCw className="h-4 w-4 mr-1" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Empty state
  if (!data || data.count === 0 || processedData.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-green-500" />
                Unhealthy Sources Analysis
              </CardTitle>
              <CardDescription>
                No unhealthy sources found • All systems operating within thresholds
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger className="w-20">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1h">1H</SelectItem>
                  <SelectItem value="6h">6H</SelectItem>
                  <SelectItem value="24h">24H</SelectItem>
                  <SelectItem value="7d">7D</SelectItem>
                </SelectContent>
              </Select>
              <Button variant="outline" size="sm" onClick={fetchUnhealthySources}>
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center h-48 text-center">
            <div className="bg-green-100 p-4 rounded-full mb-4">
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">All Systems Healthy!</h3>
            <p className="text-gray-600 mb-4">
              No sources are exceeding the alarm threshold in the selected time range.
            </p>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setTimeRange('7d')}>
                Try 7 Days Range
              </Button>
              <Button variant="outline" onClick={() => fetchUnhealthySources(true)}>
                Show All Historical Data
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Unhealthy Sources Analysis
            </CardTitle>
            <CardDescription>
              Alarm sources exceeding 10 hits per 10-minute window • Top {processedData.length} sources shown
              {data?.isHistoricalData && (
                <div className="text-amber-600 text-sm mt-1">
                  📅 {data.note}
                </div>
              )}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Select value={topLimit.toString()} onValueChange={(value) => setTopLimit(parseInt(value))}>
              <SelectTrigger className="w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">Top 10</SelectItem>
                <SelectItem value="20">Top 20</SelectItem>
                <SelectItem value="50">Top 50</SelectItem>
              </SelectContent>
            </Select>
            <Select value={sortBy} onValueChange={(value) => setSortBy(value as any)}>
              <SelectTrigger className="w-28">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="hits">By Hits</SelectItem>
                <SelectItem value="alphabetical">A-Z</SelectItem>
              </SelectContent>
            </Select>
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-20">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1h">1H</SelectItem>
                <SelectItem value="6h">6H</SelectItem>
                <SelectItem value="24h">24H</SelectItem>
                <SelectItem value="7d">7D</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="outline" size="sm" onClick={fetchUnhealthySources}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="h-[500px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 40, bottom: 120 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="source"
                angle={-45}
                textAnchor="end"
                height={100}
                tick={{ fontSize: 10 }}
                interval={0}
                tickFormatter={(value) => {
                  // Truncate long source names for better readability
                  if (value.length > 15) {
                    return value.substring(0, 12) + '...';
                  }
                  return value;
                }}
              />
              <YAxis 
                label={{ value: 'Total Hits', angle: -90, position: 'insideLeft' }}
                tick={{ fontSize: 12 }}
              />
              
              {/* Threshold line at 10 */}
              <ReferenceLine 
                y={10} 
                stroke="#ef4444" 
                strokeDasharray="5 5" 
                strokeWidth={2}
                label={{ value: "Unhealthy Threshold (10)", position: "topLeft", fill: "#ef4444", fontSize: 11 }}
              />
              
              <Tooltip content={<CustomTooltip />} />
              
              <Bar dataKey="totalHits" radius={[4, 4, 0, 0]} maxBarSize={60}>
                {processedData.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={getPriorityColor(entry.priority, entry.totalHits)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        {/* Legend and Summary */}
        <div className="mt-4 space-y-4">
          <div className="flex items-center justify-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span>High Priority (25+ hits)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-orange-500 rounded"></div>
              <span>Medium Priority (15-24 hits)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-yellow-500 rounded"></div>
              <span>Low Priority (10-14 hits)</span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
            <div className="bg-red-50 p-3 rounded-lg">
              <div className="text-red-800 font-semibold">Total Sources</div>
              <div className="text-2xl font-bold text-red-900">{processedData.length}</div>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg">
              <div className="text-orange-800 font-semibold">Total Incidents</div>
              <div className="text-2xl font-bold text-orange-900">
                {processedData.reduce((sum, item) => sum + item.incidents, 0)}
              </div>
            </div>
            <div className="bg-yellow-50 p-3 rounded-lg">
              <div className="text-yellow-800 font-semibold">Total Hits</div>
              <div className="text-2xl font-bold text-yellow-900">
                {processedData.reduce((sum, item) => sum + item.totalHits, 0)}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default UnhealthySourcesBarChart;

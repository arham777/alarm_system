import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Calendar, Clock, AlertTriangle, Filter, Download, Zap } from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  BarChart,
  Bar
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
}

interface UnhealthySourcesChartProps {
  className?: string;
}

const UnhealthySourcesChart: React.FC<UnhealthySourcesChartProps> = ({ className }) => {
  const [data, setData] = useState<UnhealthySourcesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartType, setChartType] = useState<'timeline' | 'bar'>('timeline');
  const [selectedPriority, setSelectedPriority] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [orientation, setOrientation] = useState<'horizontal' | 'vertical'>('horizontal'); // horizontal: X=time, Y=source; vertical: X=source, Y=time

  useEffect(() => {
    fetchUnhealthySources();
  }, [timeRange]);

  const fetchUnhealthySources = async () => {
    try {
      setLoading(true);
      setError(null);
      
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

      console.log(`Fetching unhealthy sources from ${startTime.toISOString()} to ${endTime.toISOString()}`);
      
      const { fetchUnhealthySources } = await import('../api/plantHealth');
      const result = await fetchUnhealthySources(
        startTime.toISOString(),
        endTime.toISOString()
      );
      
      console.log('Unhealthy sources API response:', result);
      setData(result);
      
      // Log the result for debugging
      if (result && result.count === 0) {
        console.log('No unhealthy sources found in the selected time range');
      } else if (result && result.count > 0) {
        console.log(`Found ${result.count} unhealthy sources`);
      }
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch unhealthy sources data';
      setError(errorMessage);
      console.error('Error fetching unhealthy sources:', err);
      
      // Try to provide more helpful error information
      if (errorMessage.includes('404')) {
        setError('Unhealthy sources endpoint not found. Please check if the backend server is running.');
      } else if (errorMessage.includes('500')) {
        setError('Server error while processing unhealthy sources. Please try again later.');
      }
    } finally {
      setLoading(false);
    }
  };

  // Filter data based on selected priority
  const filteredRecords = data?.records.filter(record => 
    selectedPriority === 'all' || record.priority === selectedPriority
  ) || [];

  // Prepare data for timeline scatter chart
  const timelineData = filteredRecords.map((record, index) => ({
    x: new Date(record.event_time).getTime(),
    y: record.source,
    hits: record.hits,
    over_by: record.over_by,
    rate_per_min: record.rate_per_min,
    priority: record.priority || 'Medium',
    description: record.description || 'No description',
    location_tag: record.location_tag || 'Unknown',
    condition: record.condition || 'Unknown',
    sourceIndex: [...new Set(filteredRecords.map(r => r.source))].indexOf(record.source),
    id: index
  }));

  // Prepare data for bar chart (sources by total hits)
  const sourceHitsData = filteredRecords.reduce((acc, record) => {
    const existing = acc.find(item => item.source === record.source);
    if (existing) {
      existing.totalHits += record.hits;
      existing.incidents += 1;
      existing.maxHits = Math.max(existing.maxHits, record.hits);
    } else {
      acc.push({
        source: record.source,
        totalHits: record.hits,
        incidents: 1,
        maxHits: record.hits,
        avgHits: record.hits
      });
    }
    return acc;
  }, [] as Array<{source: string, totalHits: number, incidents: number, maxHits: number, avgHits: number}>)
  .map(item => ({
    ...item,
    avgHits: Math.round(item.totalHits / item.incidents * 10) / 10
  }))
  .sort((a, b) => b.totalHits - a.totalHits)
  .slice(0, 20); // Top 20 sources

  // Get unique priorities for filter
  const priorities = ['all', ...new Set(filteredRecords.map(r => r.priority).filter(Boolean))];

  // Color mapping for priorities
  const getPriorityColor = (priority: string) => {
    switch (priority?.toLowerCase()) {
      case 'high': return '#ef4444';
      case 'medium': return '#f59e0b';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  // Custom tooltip for timeline chart
  const TimelineTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border rounded-lg shadow-lg max-w-sm">
          <div className="font-semibold text-gray-900 mb-2">{data.y}</div>
          <div className="space-y-1 text-sm">
            <div><span className="font-medium">Time:</span> {new Date(data.x).toLocaleString()}</div>
            <div><span className="font-medium">Hits:</span> {data.hits} (Threshold: 10)</div>
            <div><span className="font-medium">Over by:</span> {data.over_by} ({((data.over_by/10)*100).toFixed(1)}%)</div>
            <div><span className="font-medium">Rate:</span> {data.rate_per_min}/min</div>
            <div><span className="font-medium">Priority:</span> 
              <Badge variant="outline" className="ml-1" style={{borderColor: getPriorityColor(data.priority)}}>
                {data.priority}
              </Badge>
            </div>
            <div><span className="font-medium">Location:</span> {data.location_tag}</div>
            <div><span className="font-medium">Condition:</span> {data.condition}</div>
            {data.description !== 'No description' && (
              <div><span className="font-medium">Description:</span> {data.description}</div>
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
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p>Loading unhealthy sources...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Show empty state when no data is found
  if (!loading && !error && (!data || data.count === 0 || filteredRecords.length === 0)) {
    return (
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-green-500" />
                Unhealthy Sources Timeline
              </CardTitle>
              <CardDescription>
                No unhealthy sources found in the selected time range • All systems healthy!
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger className="w-24">
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
                <Clock className="h-4 w-4 mr-1" />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <div className="bg-green-100 p-4 rounded-full mb-4">
              <Zap className="h-8 w-8 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">All Systems Healthy!</h3>
            <p className="text-gray-600 mb-4">
              No sources are exceeding the 10 alarms per 10-minute threshold in the selected time range.
            </p>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => setTimeRange('7d')}>
                Try 7 Days
              </Button>
              <Button variant="outline" onClick={fetchUnhealthySources}>
                Refresh Data
              </Button>
            </div>
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
            <p>{error}</p>
            <Button onClick={fetchUnhealthySources} className="mt-2" variant="outline">
              Retry
            </Button>
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
              <Zap className="h-5 w-5 text-red-500" />
              Unhealthy Sources Timeline
            </CardTitle>
            <CardDescription>
              Sources exceeding 10 alarms per 10-minute window • {filteredRecords.length} incidents found
              {data && data.count === 0 && (
                <span className="text-green-600 ml-2">• All systems healthy!</span>
              )}
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Select value={timeRange} onValueChange={setTimeRange}>
              <SelectTrigger className="w-24">
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
              <Clock className="h-4 w-4 mr-1" />
              Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Filters */}
          <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <span className="text-sm font-medium">Filters:</span>
            </div>
            <Select value={selectedPriority} onValueChange={setSelectedPriority}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Priority" />
              </SelectTrigger>
              <SelectContent>
                {priorities.map(priority => (
                  <SelectItem key={priority} value={priority}>
                    {priority === 'all' ? 'All Priorities' : priority}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <div className="flex items-center gap-2 ml-auto">
              <span className="text-sm text-gray-600">Chart Type:</span>
              <Tabs value={chartType} onValueChange={(value) => setChartType(value as any)}>
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="timeline">Timeline</TabsTrigger>
                  <TabsTrigger value="bar">Top Sources</TabsTrigger>
                </TabsList>
              </Tabs>
              <div className="ml-4 flex items-center gap-2">
                <span className="text-sm text-gray-600">Orientation:</span>
                <Tabs value={orientation} onValueChange={(value) => setOrientation(value as any)}>
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="horizontal">Time →</TabsTrigger>
                    <TabsTrigger value="vertical">Time ↑</TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>
            </div>
          </div>

          {/* Charts */}
          <Tabs value={chartType} onValueChange={(value) => setChartType(value as any)}>
            <TabsContent value="timeline" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  {orientation === 'horizontal' ? (
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 120 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="number"
                        dataKey="x"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(value) => new Date(value).toLocaleString()}
                        angle={-30}
                        textAnchor="end"
                        height={60}
                      />
                      <YAxis
                        type="category"
                        dataKey="y"
                        width={110}
                        tick={{ fontSize: 12 }}
                      />
                      <ZAxis dataKey="hits" range={[60, 300]} />
                      <Tooltip content={<TimelineTooltip />} />
                      <Scatter data={timelineData}>
                        {timelineData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getPriorityColor(entry.priority)} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  ) : (
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 120 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        type="category"
                        dataKey="y"
                        tick={{ fontSize: 11 }}
                        angle={-30}
                        textAnchor="end"
                        height={60}
                      />
                      <YAxis
                        type="number"
                        dataKey="x"
                        domain={['dataMin', 'dataMax']}
                        tickFormatter={(value) => new Date(value).toLocaleString()}
                        width={150}
                      />
                      <ZAxis dataKey="hits" range={[60, 300]} />
                      <Tooltip content={<TimelineTooltip />} />
                      <Scatter data={timelineData}>
                        {timelineData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={getPriorityColor(entry.priority)} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  )}
                </ResponsiveContainer>
              </div>
              <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg">
                <div className="font-medium mb-1">How to read this chart:</div>
                <ul className="space-y-1">
                  <li>• <strong>X-axis:</strong> Event time (when the unhealthy period occurred)</li>
                  <li>• <strong>Y-axis:</strong> Source names (alarm sources)</li>
                  <li>• <strong>Dot size:</strong> Number of hits (larger = more alarms in 10 minutes)</li>
                  <li>• <strong>Dot color:</strong> Priority level (Red=High, Yellow=Medium, Green=Low)</li>
                  <li>• <strong>Hover:</strong> Shows detailed information about each incident</li>
                </ul>
              </div>
            </TabsContent>

            <TabsContent value="bar" className="space-y-4">
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={sourceHitsData} margin={{ top: 20, right: 30, bottom: 60, left: 100 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="source"
                      angle={-45}
                      textAnchor="end"
                      height={80}
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis />
                    <Tooltip 
                      formatter={(value, name) => [value, name === 'totalHits' ? 'Total Hits' : name]}
                      labelFormatter={(label) => `Source: ${label}`}
                    />
                    <Legend />
                    <Bar dataKey="totalHits" fill="#ef4444" name="Total Hits" />
                    <Bar dataKey="incidents" fill="#f59e0b" name="Incidents" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg">
                <div className="font-medium mb-1">Top unhealthy sources by total alarm hits:</div>
                <ul className="space-y-1">
                  <li>• <strong>Red bars:</strong> Total hits across all incidents</li>
                  <li>• <strong>Yellow bars:</strong> Number of separate 10-minute incidents</li>
                  <li>• Sources are ranked by total hits (most problematic first)</li>
                </ul>
              </div>
            </TabsContent>
          </Tabs>

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
            <div className="bg-red-50 p-3 rounded-lg">
              <div className="text-red-800 font-semibold">Total Incidents</div>
              <div className="text-2xl font-bold text-red-900">{filteredRecords.length}</div>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg">
              <div className="text-orange-800 font-semibold">Unique Sources</div>
              <div className="text-2xl font-bold text-orange-900">
                {new Set(filteredRecords.map(r => r.source)).size}
              </div>
            </div>
            <div className="bg-yellow-50 p-3 rounded-lg">
              <div className="text-yellow-800 font-semibold">Total Hits</div>
              <div className="text-2xl font-bold text-yellow-900">
                {filteredRecords.reduce((sum, r) => sum + r.hits, 0)}
              </div>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg">
              <div className="text-blue-800 font-semibold">Avg Hits/Incident</div>
              <div className="text-2xl font-bold text-blue-900">
                {filteredRecords.length > 0 
                  ? Math.round(filteredRecords.reduce((sum, r) => sum + r.hits, 0) / filteredRecords.length * 10) / 10
                  : 0
                }
              </div>
            </div>
          </div>

          {/* Debug Information - Remove in production */}
          {process.env.NODE_ENV === 'development' && (
            <div className="mt-4 p-3 bg-gray-100 rounded-lg text-xs">
              <div className="font-semibold mb-2">Debug Info:</div>
              <div>API Response Count: {data?.count || 'N/A'}</div>
              <div>Raw Records Length: {data?.records?.length || 0}</div>
              <div>Filtered Records Length: {filteredRecords.length}</div>
              <div>Selected Priority: {selectedPriority}</div>
              <div>Time Range: {timeRange}</div>
              <div>Loading: {loading.toString()}</div>
              <div>Error: {error || 'None'}</div>
              {data?.records?.length > 0 && (
                <div className="mt-2">
                  <div>Sample Record:</div>
                  <pre className="text-xs bg-white p-2 rounded mt-1 overflow-auto">
                    {JSON.stringify(data.records[0], null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default UnhealthySourcesChart;

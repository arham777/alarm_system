import { TrendingUp, TrendingDown, Database, FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PlantHealthMetrics } from '@/types/dashboard';

interface InsightCardsProps {
  metrics: PlantHealthMetrics;
  isLoading?: boolean;
}

export function InsightCards({ metrics, isLoading = false }: InsightCardsProps) {
  const cards = [
    {
      title: 'Healthy Sources',
      value: `${metrics.healthy_percentage.toFixed(1)}%`,
      description: 'Sources within normal range',
      icon: TrendingUp,
      trend: 'positive' as const,
    },
    {
      title: 'Unhealthy Sources',
      value: `${metrics.unhealthy_percentage.toFixed(1)}%`,
      description: 'Sources requiring attention',
      icon: TrendingDown,
      trend: 'negative' as const,
    },
    {
      title: 'Total Sources',
      value: metrics.total_sources.toLocaleString(),
      description: 'Active monitoring points',
      icon: Database,
      trend: 'neutral' as const,
    },
    {
      title: 'Total Files',
      value: metrics.total_files.toLocaleString(),
      description: 'Files processed',
      icon: FileText,
      trend: 'neutral' as const,
    },
  ];

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <Card key={i} className="shadow-metric-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <div className="h-4 w-24 bg-muted animate-pulse rounded" />
              <div className="h-4 w-4 bg-muted animate-pulse rounded" />
            </CardHeader>
            <CardContent>
              <div className="h-8 w-16 bg-muted animate-pulse rounded mb-2" />
              <div className="h-3 w-32 bg-muted animate-pulse rounded" />
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card) => {
        const Icon = card.icon;
        return (
          <Card key={card.title} className="shadow-metric-card bg-dashboard-metric-card-bg">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {card.title}
              </CardTitle>
              <Icon 
                className={`h-4 w-4 ${
                  card.trend === 'positive' 
                    ? 'text-green-600' 
                    : card.trend === 'negative' 
                    ? 'text-red-600' 
                    : 'text-muted-foreground'
                }`} 
              />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-foreground mb-1">
                {card.value}
              </div>
              <p className="text-xs text-muted-foreground">
                {card.description}
              </p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
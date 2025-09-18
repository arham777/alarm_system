import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageShell } from '@/components/dashboard/PageShell';
import { PlantSelector } from '@/components/dashboard/PlantSelector';
import { InsightCards } from '@/components/dashboard/InsightCards';
import { UnhealthyBarChart } from '@/components/dashboard/UnhealthyBarChart';
import { ErrorState } from '@/components/dashboard/ErrorState';
import UnhealthySourcesChart from '@/components/UnhealthySourcesChart';
import UnhealthySourcesBarChart from '@/components/UnhealthySourcesBarChart';
import { useAuth } from '@/hooks/useAuth';
import { usePlantHealth } from '@/hooks/usePlantHealth';
import { Plant } from '@/types/dashboard';
import { fetchPlants } from '@/api/plantHealth';


// Default plant used before API loads
const defaultPlant: Plant = { id: 'pvcI', name: 'PVC-I', status: 'active' };

export default function DashboardPage() {
  const navigate = useNavigate();
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  const [selectedPlant, setSelectedPlant] = useState<Plant>(defaultPlant);
  const [plants, setPlants] = useState<Plant[]>([]);
  const [plantsLoading, setPlantsLoading] = useState<boolean>(true);
  const [topN, setTopN] = useState<1 | 3>(1);
  
  const { 
    data, 
    isLoading, 
    error, 
    refetch, 
    isFetching 
  } = usePlantHealth(selectedPlant.id, topN);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/signin', { replace: true });
    }
  }, [isAuthenticated, authLoading, navigate]);

  // Load plants from backend
  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        setPlantsLoading(true);
        const list = await fetchPlants();
        if (!mounted) return;
        setPlants(list);
        const preferred = list.find(p => p.status === 'active') || list[0];
        if (preferred && preferred.id !== selectedPlant.id) {
          setSelectedPlant(preferred);
        }
      } finally {
        if (mounted) setPlantsLoading(false);
      }
    }
    load();
    return () => { mounted = false; };
  }, []);

  if (authLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="h-8 w-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null; // Will redirect via useEffect
  }

  const handleRefresh = () => {
    refetch();
  };

  const handlePlantChange = (plant: Plant) => {
    setSelectedPlant(plant);
  };

  const handleTopNChange = (value: 1 | 3) => {
    setTopN(value);
  };

  if (error) {
    return (
      <PageShell>
        <ErrorState
          title="Dashboard Error"
          description="Failed to load plant health data. Please check your connection and try again."
          onRetry={handleRefresh}
          isRetrying={isFetching}
        />
      </PageShell>
    );
  }

  return (
    <PageShell
      onRefresh={handleRefresh}
      isRefreshing={isFetching}
      lastUpdated={data?.metrics.last_updated}
    >
      <div className="space-y-6">
        {/* Plant Selector */}
        <div className="flex items-center justify-between">
          <PlantSelector
            plants={plants}
            selectedPlant={selectedPlant}
            onPlantChange={handlePlantChange}
            disabled={plantsLoading || plants.length <= 1}
          />
        </div>

        {/* Insight Cards */}
        <InsightCards
          metrics={data?.metrics || {
            healthy_percentage: 0,
            unhealthy_percentage: 0,
            total_sources: 0,
            total_files: 0,
            last_updated: '',
          }}
          isLoading={isLoading}
        />

        {/* Charts Section */}
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <UnhealthyBarChart
              data={data?.unhealthyBars || []}
              threshold={10}
              topN={topN}
              onTopNChange={handleTopNChange}
              isLoading={isLoading}
            />

            <UnhealthySourcesChart />
          </div>
          
          {/* New Simple Bar Chart */}
          <UnhealthySourcesBarChart />
        </div>
      </div>
    </PageShell>
  );
}
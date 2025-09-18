import { PlantHealthResponse } from '@/types/dashboard';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function fetchPlantHealth(
  plantId: string = 'pvcI',
  binSize: string = '10T',
  alarmThreshold: number = 10
): Promise<PlantHealthResponse> {
  const url = new URL(`${API_BASE_URL}/${plantId}-health/overall`);
  url.searchParams.set('bin_size', binSize);
  url.searchParams.set('alarm_threshold', alarmThreshold.toString());

  try {
    const response = await fetch(url.toString());
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    // Fallback mock data for development
    console.warn('Failed to fetch from API, using mock data:', error);
    return getMockPlantHealthData();
  }
}

function getMockPlantHealthData(): PlantHealthResponse {
  return {
    metrics: {
      healthy_percentage: 87.5,
      unhealthy_percentage: 12.5,
      total_sources: 24,
      total_files: 1547,
      last_updated: new Date().toISOString(),
    },
    unhealthy_bins: [
      {
        source: 'sensor-01',
        hits: 15,
        threshold: 10,
        over_by: 5,
        bin_start: '2024-01-15T10:00:00Z',
        bin_end: '2024-01-15T10:10:00Z',
      },
      {
        source: 'sensor-03',
        hits: 23,
        threshold: 10,
        over_by: 13,
        bin_start: '2024-01-15T10:10:00Z',
        bin_end: '2024-01-15T10:20:00Z',
      },
      {
        source: 'sensor-01',
        hits: 12,
        threshold: 10,
        over_by: 2,
        bin_start: '2024-01-15T10:20:00Z',
        bin_end: '2024-01-15T10:30:00Z',
      },
      {
        source: 'sensor-07',
        hits: 18,
        threshold: 10,
        over_by: 8,
        bin_start: '2024-01-15T10:15:00Z',
        bin_end: '2024-01-15T10:25:00Z',
      },
    ],
  };
}
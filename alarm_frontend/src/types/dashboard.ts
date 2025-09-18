export interface PlantHealthMetrics {
  healthy_percentage: number;
  unhealthy_percentage: number;
  total_sources: number;
  total_files: number;
  last_updated: string;
}

export interface UnhealthyBin {
  source: string;
  hits: number;
  threshold: number;
  over_by: number;
  bin_start: string;
  bin_end: string;
}

export interface UnhealthyBar {
  id: string;
  source: string;
  hits: number;
  threshold: number;
  over_by: number;
  bin_start: string;
  bin_end: string;
}

export interface PlantHealthResponse {
  metrics: PlantHealthMetrics;
  unhealthy_bins: UnhealthyBin[];
}

export interface Plant {
  id: string;
  name: string;
  status: 'active' | 'inactive';
}
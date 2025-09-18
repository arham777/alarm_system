import { useQuery } from '@tanstack/react-query';
import { fetchPlantHealth } from '@/api/plantHealth';
import { UnhealthyBar, UnhealthyBin } from '@/types/dashboard';

function transformUnhealthyBins(
  bins: UnhealthyBin[], 
  topN: 1 | 3
): UnhealthyBar[] {
  // Group by source and find max hits per source
  const sourceMaxHits = bins.reduce((acc, bin) => {
    const current = acc[bin.source];
    if (!current || bin.hits > current.hits) {
      acc[bin.source] = bin;
    }
    return acc;
  }, {} as Record<string, UnhealthyBin>);

  // Sort by hits descending and take top N per source
  const sortedSources = Object.values(sourceMaxHits)
    .sort((a, b) => b.hits - a.hits);

  // For top N = 1, take the single worst bin per source (already done above)
  // For top N = 3, take up to 3 worst bins per source
  let result: UnhealthyBar[] = [];
  
  if (topN === 1) {
    result = sortedSources.map(bin => ({
      id: `${bin.source}-${bin.bin_start}`,
      ...bin,
    }));
  } else {
    // For top 3, get up to 3 worst bins per source
    const sourceGroups = bins.reduce((acc, bin) => {
      if (!acc[bin.source]) acc[bin.source] = [];
      acc[bin.source].push(bin);
      return acc;
    }, {} as Record<string, UnhealthyBin[]>);

    // Sort each source group by hits and take top 3
    Object.entries(sourceGroups).forEach(([source, sourceBins]) => {
      const topBins = sourceBins
        .sort((a, b) => b.hits - a.hits)
        .slice(0, 3);
      
      result.push(...topBins.map(bin => ({
        id: `${bin.source}-${bin.bin_start}`,
        ...bin,
      })));
    });

    // Sort final result by hits
    result.sort((a, b) => b.hits - a.hits);
  }

  return result;
}

export function usePlantHealth(
  plantId: string = 'pvcI',
  topN: 1 | 3 = 1,
  refetchInterval: number = 60000 // 60 seconds
) {
  return useQuery({
    queryKey: ['plant-health', plantId, topN],
    queryFn: () => fetchPlantHealth(plantId),
    refetchInterval,
    refetchIntervalInBackground: true,
    staleTime: 30000, // 30 seconds
    select: (data) => ({
      ...data,
      unhealthyBars: transformUnhealthyBins(data.unhealthy_bins, topN),
    }),
  });
}
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, List, Optional
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
from config import PVCI_FOLDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='health_monitor.log'
)
logger = logging.getLogger('health_monitor')

class HealthConfig:
    """Configuration class for health monitoring parameters"""
    def __init__(
        self,
        bin_size: str = '10T',
        alarm_threshold: int = 10,
        bins_per_day: int = 144,
        skip_rows: int = 8,
        cache_size: int = 100
    ):
        self.bin_size = bin_size
        self.alarm_threshold = alarm_threshold
        self.bins_per_day = bins_per_day
        self.skip_rows = skip_rows
        self.cache_size = cache_size

# Global configuration
DEFAULT_CONFIG = HealthConfig()

@lru_cache(maxsize=100)
def read_csv_smart(file_path: str, config: HealthConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Smart CSV reader with caching and efficient memory usage"""
    try:
        # First try to determine separator from header
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = ''.join([next(f) for _ in range(min(10, config.skip_rows + 2))])
        sep = ',' if ',' in sample else '\t'
        
        logger.info(f"Reading file {file_path} with separator '{sep}'")
        
        # Attempt to read first chunk to validate structure
        test_chunk = pd.read_csv(
            file_path,
            sep=sep,
            skiprows=config.skip_rows,
            nrows=5,
            encoding='utf-8'
        )
        
        if 'Event Time' not in test_chunk.columns or 'Source' not in test_chunk.columns:
            raise ValueError(f"Required columns 'Event Time' and 'Source' not found. Available columns: {list(test_chunk.columns)}")

        # Read in chunks to handle large files
        chunks = []
        chunk_size = 10000  # Adjust based on available memory
        
        logger.info(f"Processing {file_path} in chunks")
        
        for chunk_num, chunk in enumerate(pd.read_csv(
            file_path,
            sep=sep,
            skiprows=config.skip_rows,
            chunksize=chunk_size,
            usecols=['Event Time', 'Source'],  # Only reading Event Time and Source columns
            parse_dates=['Event Time'],
            encoding='utf-8',
            on_bad_lines='skip'  # Skip problematic lines
        )):
            logger.debug(f"Processing chunk {chunk_num + 1}")
            # Keep only the base source name and event time
            chunk_data = chunk[['Event Time', 'Source']].copy()
            chunks.append(chunk_data)
        
        if not chunks:
            logger.warning(f"No valid data found in {file_path}")
            return pd.DataFrame(columns=['Event Time', 'Source'])
            
        df = pd.concat(chunks, ignore_index=True)
        
        # Clean data
        df['Event Time'] = pd.to_datetime(df['Event Time'], errors='coerce')
        df = df.dropna(subset=['Event Time', 'Source'])
        df['Source'] = df['Source'].str.strip()
        
        logger.info(f"Successfully read {len(df)} rows from {file_path}")
        return df
        
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty file: {file_path}")
        return pd.DataFrame(columns=['Event Time', 'Source'])
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise ValueError(f"Failed to read file: {str(e)}")

def compute_file_health(
    file_path: str,
    config: HealthConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """Compute health metrics for a single file"""
    try:
        logger.info(f"Processing file: {file_path}")
        df = read_csv_smart(file_path, config)
        
        if df.empty:
            logger.info(f"No data in file: {file_path}")
            return {'health_pct': 100.0, 'message': 'No data'}
        
        # Get time range for the file
        start_time = df['Event Time'].min()
        full_bins = pd.date_range(
            start=start_time.floor('D'),
            periods=config.bins_per_day,
            freq=config.bin_size
        )
        
        # Efficient vectorized operations for all sources at once
        df.set_index('Event Time', inplace=True)
        
        # Group by source and time bin
        grouped = df.groupby([
            pd.Grouper(freq=config.bin_size),
            'Source'
        ]).size().unstack(fill_value=0)
        
        # Reindex to ensure all time bins exist
        grouped = grouped.reindex(full_bins, fill_value=0)
        
        # Calculate health scores efficiently using numpy operations
        healthy_mask = (grouped <= config.alarm_threshold)
        health_scores = healthy_mask.sum()
        
        # Derive additional masks and counts for detailed bin statistics
        unhealthy_mask = (grouped > config.alarm_threshold)
        bins_with_events_series = (grouped > 0).sum()
        healthy_bins_series = healthy_mask.sum()
        unhealthy_bins_series = unhealthy_mask.sum()
        
        # Build per-source bin statistics
        source_bin_stats: Dict[str, Any] = {}
        for src in grouped.columns:
            src_name = str(src)
            total_bins_for_source = int(config.bins_per_day)
            bins_with_events = int(bins_with_events_series[src])
            healthy_bins_count = int(healthy_bins_series[src])
            unhealthy_bins_count = int(unhealthy_bins_series[src])

            stats: Dict[str, Any] = {
                'total_bins': total_bins_for_source,
                'bins_with_events': bins_with_events,
                'healthy_bins': healthy_bins_count,
                'unhealthy_bins': unhealthy_bins_count
            }

            # Only attach details for sources that are not 100% healthy
            if unhealthy_bins_count > 0:
                bad_mask = unhealthy_mask[src]
                details = [
                    {
                        'bin_start': idx.isoformat(),
                        'count': int(grouped.at[idx, src])
                    }
                    for idx in grouped.index[bad_mask]
                ]
                stats['unhealthy_bin_details'] = details

            source_bin_stats[src_name] = stats
        
        # Summarize repeating alarm sources (sorted by number of unhealthy bins desc)
        repeating_alarm_sources = sorted(
            [
                (str(src), int(unhealthy_bins_series[src]))
                for src in grouped.columns
                if int(unhealthy_bins_series[src]) > 0
            ],
            key=lambda x: x[1],
            reverse=True
        )
        
        num_sources = int(len(health_scores))
        total_healthy = int(health_scores.sum())
        total_bins = num_sources * config.bins_per_day
        health_pct = float((total_healthy / total_bins * 100) if total_bins > 0 else 100)
        
        # Calculate additional metrics and convert to native Python types
        source_health = {
            str(source): float(health) 
            for source, health in (health_scores / config.bins_per_day * 100).items()
        }
        
        worst_sources = [
            (str(source), float(health)) 
            for source, health in sorted(
                source_health.items(),
                key=lambda x: x[1]
            )[:3]
        ]
        
        return {
            'filename': os.path.basename(file_path),
            'num_sources': num_sources,
            'health_pct': round(health_pct, 2),
            'unhealthy_count': total_bins - total_healthy,
            'source_health': source_health,
            'source_bin_stats': source_bin_stats,
            'repeating_alarm_sources': repeating_alarm_sources,
            'worst_performing_sources': worst_sources,
            'data_start': start_time.isoformat(),
            'data_end': df.index.max().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error computing health for {file_path}: {str(e)}")
        raise

def compute_overall_health(
    folder_path: str,
    config: HealthConfig = DEFAULT_CONFIG,
    max_workers: int = 4
) -> Dict[str, Any]:
    """Compute overall health metrics for all files in parallel"""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    daily_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                compute_file_health,
                os.path.join(folder_path, file),
                config
            ): file for file in files
        }
        
        for future in future_to_file:
            try:
                result = future.result()
                daily_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {future_to_file[future]}: {str(e)}")
                continue
    
    if not daily_results:
        return {
            'total_files': 0,
            'overall_health_pct': 100,
            'message': 'No files processed'
        }
    
    # Calculate overall metrics
    health_values = [r['health_pct'] for r in daily_results]
    overall_health = sum(health_values) / len(health_values)
    
    # Aggregate source health across all files
    all_source_health = {}
    for result in daily_results:
        for source, health in result.get('source_health', {}).items():
            if source not in all_source_health:
                all_source_health[source] = []
            all_source_health[source].append(health)
    
    # Calculate average health per source
    avg_source_health = {
        source: sum(health) / len(health)
        for source, health in all_source_health.items()
    }
    
    return {
        'total_files': len(files),
        'overall_health_pct': round(overall_health, 2),
        'daily_results': daily_results,
        'avg_source_health': avg_source_health,
        'worst_performing_sources': sorted(
            avg_source_health.items(),
            key=lambda x: x[1]
        )[:5],
        'processed_files': len(daily_results),
        'failed_files': len(files) - len(daily_results)
    }

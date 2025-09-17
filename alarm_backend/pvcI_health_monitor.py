import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, List, Optional
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
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
        
        try:
            # Fast path: use default C engine without parse_dates to maximize speed
            reader = pd.read_csv(
                file_path,
                sep=sep,
                skiprows=config.skip_rows,
                chunksize=chunk_size,
                usecols=['Event Time', 'Source'],
                encoding='utf-8'
            )
            for chunk_num, chunk in enumerate(reader):
                logger.debug(f"Processing chunk {chunk_num + 1}")
                chunk_data = chunk[['Event Time', 'Source']].copy()
                chunks.append(chunk_data)
        except Exception as e:
            # Fallback: use Python engine and skip bad lines if fast path fails
            logger.warning(f"Fast CSV parse failed for {file_path} with error: {str(e)}. Retrying with python engine and skipping bad lines.")
            reader = pd.read_csv(
                file_path,
                sep=sep,
                skiprows=config.skip_rows,
                chunksize=chunk_size,
                usecols=['Event Time', 'Source'],
                encoding='utf-8',
                engine='python',
                on_bad_lines='skip'
            )
            for chunk_num, chunk in enumerate(reader):
                logger.debug(f"Processing chunk {chunk_num + 1} [python engine]")
                chunk_data = chunk[['Event Time', 'Source']].copy()
                chunks.append(chunk_data)
        
        if not chunks:
            logger.warning(f"No valid data found in {file_path}")
            return pd.DataFrame(columns=['Event Time', 'Source'])
            
        df = pd.concat(chunks, ignore_index=True)
        
        # Clean data: single datetime parse after reading for speed
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

def compute_pvcI_file_health(
    file_path: str,
    config: HealthConfig = DEFAULT_CONFIG,
    include_details: bool = True
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
        
        # Build per-source bin statistics (optional for performance)
        source_bin_stats: Dict[str, Any] = {}
        # Consolidated per-source details to embed health and bin stats together
        source_details: Dict[str, Any] = {}
        if include_details:
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

                # Compute details list once; keep stats backward-compatible (only add when unhealthy>0)
                bad_mask = unhealthy_mask[src]
                details = [
                    {
                        'bin_start': idx.isoformat(),
                        'count': int(grouped.at[idx, src])
                    }
                    for idx in grouped.index[bad_mask]
                ]
                if unhealthy_bins_count > 0:
                    stats['unhealthy_bin_details'] = details

                source_bin_stats[src_name] = stats

                # Build consolidated per-source view (health + stats + optional details)
                health_pct_src = float(healthy_bins_count / total_bins_for_source * 100)
                consolidated = dict(stats)  # shallow copy
                consolidated['health_pct'] = health_pct_src
                # Always include unhealthy_bin_details key for consistent API (empty array when none)
                if 'unhealthy_bin_details' not in consolidated:
                    consolidated['unhealthy_bin_details'] = details
                source_details[src_name] = consolidated
        
        # Summarize repeating alarm sources (optional heavy metric)
        repeating_alarm_sources = []
        if include_details:
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
        
        worst_sources = []
        if include_details:
            worst_sources = [
                (str(source), float(health)) 
                for source, health in sorted(
                    source_health.items(),
                    key=lambda x: x[1]
                )[:3]
            ]
            # Enrich consolidated source_details with repeating_count and worst flags
            # repeating_count is simply the number of unhealthy bins for the source
            for src in grouped.columns:
                src_name = str(src)
                if 'source_details' in locals() and src_name in source_details:
                    source_details[src_name]['repeating_count'] = int(unhealthy_bins_series[src])
            # Mark worst performing top-3 with rank
            for rank, (src_name, _health) in enumerate(worst_sources, start=1):
                if 'source_details' in locals() and src_name in source_details:
                    source_details[src_name]['is_worst_performing'] = True
                    source_details[src_name]['worst_rank'] = rank
        
        return {
            'filename': os.path.basename(file_path),
            'num_sources': num_sources,
            'health_pct': round(health_pct, 2),
            'unhealthy_count': total_bins - total_healthy,
            'source_health': source_health,
            'source_bin_stats': source_bin_stats if include_details else None,
            # New consolidated per-source structure for inline health and bins
            'source_details': source_details if include_details else None,
            'repeating_alarm_sources': repeating_alarm_sources,
            'worst_performing_sources': worst_sources,
            'data_start': start_time.isoformat(),
            'data_end': df.index.max().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error computing health for {file_path}: {str(e)}")
        raise

def compute_pvcI_overall_health(
    folder_path: str,
    config: HealthConfig = DEFAULT_CONFIG,
    max_workers: int = 4,
    per_file_timeout: Optional[int] = None,
    include_details: bool = False
) -> Dict[str, Any]:
    """Compute overall health metrics for all files in parallel"""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    daily_results = []
    
    # If no timeout, collect as results complete
    if per_file_timeout is None:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    compute_pvcI_file_health,
                    os.path.join(folder_path, file),
                    config,
                    include_details
                ) for file in files
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    # Trim heavy per-file details to keep overall response light
                    trimmed = dict(result)
                    if 'source_bin_stats' in trimmed:
                        del trimmed['source_bin_stats']
                    daily_results.append(trimmed)
                except Exception as e:
                    logger.error(f"Error processing a file in overall health: {str(e)}")
                    continue
    else:
        # Enforce a global timeout for all files to avoid long blocking
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [
                executor.submit(
                    compute_pvcI_file_health,
                    os.path.join(folder_path, file),
                    config,
                    include_details
                ) for file in files
            ]
            done, not_done = wait(futures, timeout=per_file_timeout)
            for future in done:
                try:
                    result = future.result()
                    trimmed = dict(result)
                    if 'source_bin_stats' in trimmed:
                        del trimmed['source_bin_stats']
                    daily_results.append(trimmed)
                except Exception as e:
                    logger.error(f"Error processing a file in overall health: {str(e)}")
                    continue
            if not_done:
                logger.warning(f"Overall health timed out after {per_file_timeout}s. Skipped {len(not_done)} file(s).")
        finally:
            # Try to cancel remaining futures and return immediately
            executor.shutdown(wait=False, cancel_futures=True)
    
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

def compute_pvcI_overall_health_weighted(
    folder_path: str,
    config: HealthConfig = DEFAULT_CONFIG,
    max_workers: int = 4,
    per_file_timeout: Optional[int] = None,
    include_details: bool = False
) -> Dict[str, Any]:
    """Compute weighted overall health metrics for all files in parallel.
    The response shape matches compute_overall_health; only overall_health_pct differs (weighted).
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    daily_results = []

    # Generate daily_results exactly the same way to keep schema identical
    if per_file_timeout is None:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    compute_pvcI_file_health,
                    os.path.join(folder_path, file),
                    config,
                    include_details
                ) for file in files
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    trimmed = dict(result)
                    if 'source_bin_stats' in trimmed:
                        del trimmed['source_bin_stats']
                    daily_results.append(trimmed)
                except Exception as e:
                    logger.error(f"Error processing a file in overall health (weighted): {str(e)}")
                    continue
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = [
                executor.submit(
                    compute_pvcI_file_health,
                    os.path.join(folder_path, file),
                    config,
                    include_details
                ) for file in files
            ]
            done, not_done = wait(futures, timeout=per_file_timeout)
            for future in done:
                try:
                    result = future.result()
                    trimmed = dict(result)
                    if 'source_bin_stats' in trimmed:
                        del trimmed['source_bin_stats']
                    daily_results.append(trimmed)
                except Exception as e:
                    logger.error(f"Error processing a file in overall health (weighted): {str(e)}")
                    continue
            if not_done:
                logger.warning(f"Overall health (weighted) timed out after {per_file_timeout}s. Skipped {len(not_done)} file(s).")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    if not daily_results:
        return {
            'total_files': 0,
            'overall_health_pct': 100,
            'message': 'No files processed'
        }

    # Weighted overall: sum healthy bins / sum total bins across all processed files
    total_bins_all = 0
    total_healthy_all = 0
    for r in daily_results:
        num_sources = int(r.get('num_sources', 0))
        total_bins_file = num_sources * config.bins_per_day
        unhealthy_count = int(r.get('unhealthy_count', 0))
        healthy_bins_file = max(total_bins_file - unhealthy_count, 0)
        total_bins_all += total_bins_file
        total_healthy_all += healthy_bins_file

    overall_health_weighted = (total_healthy_all / total_bins_all * 100) if total_bins_all > 0 else 100

    # Aggregate source health across all files (same as unweighted version for schema parity)
    all_source_health = {}
    for result in daily_results:
        for source, health in result.get('source_health', {}).items():
            if source not in all_source_health:
                all_source_health[source] = []
            all_source_health[source].append(health)

    avg_source_health = {
        source: sum(health) / len(health)
        for source, health in all_source_health.items()
    }

    return {
        'total_files': len(files),
        'overall_health_pct': round(overall_health_weighted, 2),
        'daily_results': daily_results,
        'avg_source_health': avg_source_health,
        'worst_performing_sources': sorted(
            avg_source_health.items(),
            key=lambda x: x[1]
        )[:5],
        'processed_files': len(daily_results),
        'failed_files': len(files) - len(daily_results)
    }

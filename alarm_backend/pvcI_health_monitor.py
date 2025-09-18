import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any, List, Optional
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from config import PVCI_FOLDER
from pvcI_files import read_pvc_file, list_pvc_files

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
    """
    Compute health for one PVC-I day file and return a JSON-serializable dict.

    Enhancements:
      - Keeps original structure (filename, source_details, health_pct, etc.)
      - Enriches `unhealthy_bin_details` per source with:
          bin_start, bin_end, window_minutes,
          hits, threshold, over_by, over_pct, rate_per_min,
          first_event_ts, last_event_ts, unique_event_minutes,
          source, status="unhealthy"
    """
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    # -------------------- 0) Read CSV via your helper --------------------
    try:
        df = read_csv_smart(file_path, config)
    except Exception as e:
        return {
            "filename": os.path.basename(file_path),
            "num_sources": 0,
            "health_pct": 100.0,
            "unhealthy_count": 0,
            "error": f"Failed to read CSV: {e}",
            "source_details": {}
        }

    filename = os.path.basename(file_path)

    # Empty / invalid DF guard
    if df is None or len(df) == 0:
        return {
            "filename": filename,
            "num_sources": 0,
            "health_pct": 100.0,
            "unhealthy_count": 0,
            "source_details": {}
        }

    # -------------------- 1) Resolve columns --------------------
    # Timestamp column candidates commonly seen in PVC-I dumps
    ts_candidates = ("timestamp", "Timestamp", "time", "Time", "TIMESTAMP", "Event Time", "event_time")
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        # If your reader standardizes to a known name, set it here:
        # ts_col = "__ts__"
        # else: fail
        raise ValueError(f"No timestamp column found in {filename}. "
                         f"Tried {list(ts_candidates)}; got columns {list(df.columns)}")

    source_candidates = ("Source", "source", "AlarmSource", "alarm_source", "SRC", "src")
    source_col = next((c for c in source_candidates if c in df.columns), None)
    if source_col is None:
        raise ValueError(f"No source column found in {filename}. "
                         f"Tried {list(source_candidates)}; got columns {list(df.columns)}")

    # -------------------- 2) Normalize timestamps (UTC, monotonic) --------------------
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.copy()
    df["__ts__"] = ts
    df = df.dropna(subset=["__ts__", source_col]).sort_values("__ts__").reset_index(drop=True)

    if df.empty:
        return {
            "filename": filename,
            "num_sources": 0,
            "health_pct": 100.0,
            "unhealthy_count": 0,
            "source_details": {}
        }

    # -------------------- 3) Window config --------------------
    window_minutes = getattr(config, "window_minutes", 10) if hasattr(config, "window_minutes") else 10
    alarm_threshold = getattr(config, "alarm_threshold", 10) if hasattr(config, "alarm_threshold") else 10

    bin_freq = f"{int(window_minutes)}T"
    off = to_offset(bin_freq)

    first_ts = df["__ts__"].min()
    last_ts = df["__ts__"].max()

    # Full grid covering [floor(first), ceil(last)+freq) at 10-min steps
    grid_start = first_ts.floor(bin_freq)
    grid_end_excl = last_ts.ceil(bin_freq) + off
    full_bins = pd.date_range(start=grid_start, end=grid_end_excl, freq=bin_freq, inclusive="left")
    if len(full_bins) == 0:
        full_bins = pd.DatetimeIndex([first_ts.floor(bin_freq)])

    # -------------------- 4) Count hits per source per bin --------------------
    df["__bin_start__"] = df["__ts__"].dt.floor(bin_freq)
    counts = (
        df.groupby([source_col, "__bin_start__"])
          .size()
          .rename("hits")
          .reset_index()
    )

    grouped = (
        counts.pivot(index="__bin_start__", columns=source_col, values="hits")
              .reindex(full_bins, fill_value=0)
              .sort_index()
    )
    grouped.index.name = "__bin_start__"

    total_bins = int(len(grouped.index))
    if total_bins == 0:
        return {
            "filename": filename,
            "num_sources": int(len(grouped.columns)),
            "health_pct": 100.0,
            "unhealthy_count": 0,
            "source_details": {}
        }

    # -------------------- 5) Per-source health & enriched unhealthy details --------------------
    source_details: Dict[str, Any] = {}
    unhealthy_total = 0

    def _window_end(t0):
        return t0 + off

    for src in grouped.columns:
        series = grouped[src].fillna(0).astype(int)

        unhealthy_mask = series > alarm_threshold  # strictly greater than 10 = unhealthy
        unhealthy_bins_count = int(unhealthy_mask.sum())
        unhealthy_total += unhealthy_bins_count

        bins_with_events = int((series > 0).sum())
        healthy_bins = int(total_bins - unhealthy_bins_count)
        health_pct = round((healthy_bins / total_bins) * 100.0, 6)

        # Details (only if requested)
        details = []
        if include_details and unhealthy_bins_count > 0:
            # raw rows for this source (for diagnostics)
            src_df = df.loc[df[source_col] == src, ["__ts__"]].copy()
            src_df["__minute__"] = src_df["__ts__"].dt.floor("min")

            bad_idx = grouped.index[unhealthy_mask]
            for t0 in bad_idx:
                t1 = _window_end(t0)
                hits = int(series.loc[t0])
                over_by = hits - alarm_threshold
                over_pct = round((over_by / alarm_threshold) * 100.0, 6) if alarm_threshold > 0 else None
                rate_per_min = round(hits / float(window_minutes), 6) if window_minutes else None

                wmask = (src_df["__ts__"] >= t0) & (src_df["__ts__"] < t1)
                sub = src_df.loc[wmask]
                first_event = sub["__ts__"].min() if not sub.empty else pd.NaT
                last_event = sub["__ts__"].max() if not sub.empty else pd.NaT
                uniq_min = int(sub["__minute__"].nunique()) if not sub.empty else 0

                details.append({
                    "source": src,
                    "status": "unhealthy",
                    "bin_start": t0.isoformat(),
                    "bin_end": t1.isoformat(),
                    "window_minutes": int(window_minutes),

                    "hits": hits,
                    "threshold": int(alarm_threshold),
                    "over_by": int(over_by),
                    "over_pct": over_pct,
                    "rate_per_min": rate_per_min,

                    "first_event_ts": None if pd.isna(first_event) else first_event.isoformat(),
                    "last_event_ts":  None if pd.isna(last_event)  else last_event.isoformat(),
                    "unique_event_minutes": uniq_min
                })

        source_details[str(src)] = {
            "total_bins": int(total_bins),
            "bins_with_events": int(bins_with_events),
            "healthy_bins": int(healthy_bins),
            "unhealthy_bins": int(unhealthy_bins_count),
            "health_pct": health_pct,
            "unhealthy_bin_details": details
        }

    # -------------------- 6) Overall file health --------------------
    if len(grouped.columns) > 0:
        overall_health = sum(d["health_pct"] for d in source_details.values()) / float(len(grouped.columns))
    else:
        overall_health = 100.0

    return {
        "filename": filename,
        "num_sources": int(len(grouped.columns)),
        "health_pct": round(overall_health, 6),
        "unhealthy_count": int(unhealthy_total),
        "source_details": source_details
    }
def compute_pvcI_overall_health(
    folder_path: str,
    config: HealthConfig = DEFAULT_CONFIG,
    max_workers: int = 4,
    per_file_timeout: float = 20.0,
    include_details: bool = True,
    limit_unhealthy_per_source: "Optional[int]" = None,
) -> "Dict[str, Any]":
    """
    Robust overall health calculator with timeouts, payload control and rich errors.
    """
    # ---- local imports so missing symbols can't break module import ----
    import os, datetime as dt, traceback
    from typing import Dict, Any, List, Optional
    from concurrent.futures import ThreadPoolExecutor, wait

    try:
        # 0) discover CSV files
        try:
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            entries = sorted(
                [e.name for e in os.scandir(folder_path) if e.is_file() and e.name.lower().endswith(".csv")]
            )
        except Exception as e:
            # return minimal (not raise) so API still answers
            return {
                "plant_folder": folder_path,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "overall": {"health_pct_simple": 100.0, "health_pct_weighted": 100.0},
                "files": [],
                "per_source": {},
                "errors": [f"Failed to list folder: {e}"],
            }

        file_paths = [os.path.join(folder_path, f) for f in entries]

        # 1) per-file compute (parallel with global timeout)
        file_results: List[Dict[str, Any]] = []
        errors: List[str] = []

        if max_workers and max_workers > 1 and file_paths:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        compute_pvcI_file_health,
                        p,
                        config,
                        include_details
                    ) for p in file_paths
                ]
                done, not_done = wait(futures, timeout=per_file_timeout)
                for fut in done:
                    try:
                        file_results.append(fut.result())
                    except Exception as e:
                        errors.append(str(e))
                for fut in not_done:
                    fut.cancel()
                if not_done:
                    errors.append(f"Timed out after {per_file_timeout:.1f}s; skipped {len(not_done)} file(s).")
        else:
            for p in file_paths:
                try:
                    file_results.append(
                        compute_pvcI_file_health(p, config=config, include_details=include_details)
                    )
                except Exception as e:
                    errors.append(f"Error processing {os.path.basename(p)}: {e}")

        if not file_results:
            return {
                "plant_folder": folder_path,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "overall": {"health_pct_simple": 100.0, "health_pct_weighted": 100.0},
                "files": [],
                "per_source": {},
                "errors": errors,
            }

        # 2) per-file summaries
        files_summary: List[Dict[str, Any]] = []
        total_bins_all = 0
        healthy_bins_all = 0
        total_sources_all = 0

        for r in file_results:
            filename = r.get("filename", "unknown")
            num_sources = int(r.get("num_sources", 0) or 0)
            health_pct = float(r.get("health_pct", 100.0) or 100.0)
            source_details = r.get("source_details") or {}

            file_total_bins = 0
            file_healthy_bins = 0
            file_unhealthy_bins = 0
            for sd in source_details.values():
                tb = int(sd.get("total_bins", 0) or 0)
                hb = int(sd.get("healthy_bins", 0) or 0)
                ub = int(sd.get("unhealthy_bins", 0) or 0)
                file_total_bins += tb
                file_healthy_bins += hb
                file_unhealthy_bins += ub

            files_summary.append({
                "filename": filename,
                "num_sources": num_sources,
                "health_pct": round(health_pct, 6),
                "total_bins": file_total_bins,
                "healthy_bins": file_healthy_bins,
                "unhealthy_bins": file_unhealthy_bins,
            })

            total_bins_all += file_total_bins
            healthy_bins_all += file_healthy_bins
            total_sources_all += num_sources

        # 3) plant level
        simple_avg = sum(f["health_pct"] for f in files_summary) / float(len(files_summary))
        weighted_pct = (healthy_bins_all / float(total_bins_all) * 100.0) if total_bins_all > 0 else 100.0

        overall = {
            "health_pct_simple": round(simple_avg, 6),
            "health_pct_weighted": round(weighted_pct, 6),
            "totals": {
                "files": len(files_summary),
                "sources": int(total_sources_all),
                "total_bins": int(total_bins_all),
                "healthy_bins": int(healthy_bins_all),
                "unhealthy_bins": int(total_bins_all - healthy_bins_all),
            }
        }

        # 4) per-source aggregation
        per_source: Dict[str, Dict[str, Any]] = {}
        if include_details:
            for r in file_results:
                src_map = r.get("source_details") or {}
                fname = r.get("filename", "unknown")
                for src, sd in src_map.items():
                    agg = per_source.setdefault(src, {
                        "files_touched": 0, "total_bins": 0, "healthy_bins": 0, "unhealthy_bins": 0,
                        "health_pct": 100.0, "unhealthy_bin_details": [],
                    })
                    agg["files_touched"] += 1
                    agg["total_bins"] += int(sd.get("total_bins", 0) or 0)
                    agg["healthy_bins"] += int(sd.get("healthy_bins", 0) or 0)
                    agg["unhealthy_bins"] += int(sd.get("unhealthy_bins", 0) or 0)

                    for d in (sd.get("unhealthy_bin_details") or []):
                        it = dict(d); it["filename"] = fname
                        agg["unhealthy_bin_details"].append(it)

            for src, agg in per_source.items():
                tb = int(agg["total_bins"])
                hb = int(agg["healthy_bins"])
                agg["health_pct"] = round((hb / float(tb)) * 100.0, 6) if tb > 0 else 100.0

                dets = agg.get("unhealthy_bin_details")
                if dets:
                    dets.sort(key=lambda x: (int(x.get("over_by", 0) or 0), x.get("bin_start", "")), reverse=True)
                    if limit_unhealthy_per_source and limit_unhealthy_per_source > 0:
                        agg["unhealthy_bin_details"] = dets[:limit_unhealthy_per_source]
        else:
            for r in file_results:
                src_map = r.get("source_details") or {}
                for src, sd in src_map.items():
                    agg = per_source.setdefault(src, {
                        "files_touched": 0, "total_bins": 0, "healthy_bins": 0, "unhealthy_bins": 0,
                        "health_pct": 100.0
                    })
                    agg["files_touched"] += 1
                    agg["total_bins"] += int(sd.get("total_bins", 0) or 0)
                    agg["healthy_bins"] += int(sd.get("healthy_bins", 0) or 0)
                    agg["unhealthy_bins"] += int(sd.get("unhealthy_bins", 0) or 0)

            for src, agg in per_source.items():
                tb = int(agg["total_bins"]); hb = int(agg["healthy_bins"])
                agg["health_pct"] = round((hb / float(tb)) * 100.0, 6) if tb > 0 else 100.0

        # 5) response
        return {
            "plant_folder": folder_path,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "overall": overall,
            "files": files_summary,
            "per_source": per_source,
            "errors": errors,
        }

    except Exception:
        # bubble a readable stack up to the endpoint
        raise ValueError("overall_health_failure:\n" + traceback.format_exc())





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



def compute_pvcI_unhealthy_sources(
    folder_path: str,
    config: "HealthConfig",
    max_workers: int = 4,
    per_file_timeout: float = 20.0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns only the unhealthy bins with full metadata for plotting.
    Uses compute_pvcI_overall_health() to get unhealthy bins, then reads CSV files for metadata.
    """
    result = compute_pvcI_overall_health(folder_path, config, max_workers, per_file_timeout)
    per_source = result.get("per_source", {})

    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None

    unhealthy_records: List[Dict[str, Any]] = []

    for source, details in per_source.items():
        for bin_detail in details.get("unhealthy_bin_details", []):
            bin_start_str = bin_detail["bin_start"]
            bin_start = datetime.fromisoformat(bin_start_str.replace("Z", "+00:00"))

            # Filter by time range
            if start_dt and bin_start < start_dt:
                continue
            if end_dt and bin_start > end_dt:
                continue

            # Extract metadata by reading the CSV file directly
            file_path = os.path.join(folder_path, bin_detail["filename"])
            metadata = _extract_metadata_from_csv(
                file_path, source, bin_detail["bin_start"], bin_detail["bin_end"]
            )

            unhealthy_records.append({
                "event_time": bin_detail["bin_start"],
                "bin_end": bin_detail["bin_end"],
                "source": source,
                "hits": bin_detail["hits"],
                "threshold": bin_detail["threshold"],
                "over_by": bin_detail["over_by"],
                "rate_per_min": bin_detail["rate_per_min"],
                **metadata
            })

    return {"count": len(unhealthy_records), "records": unhealthy_records}


def _default_metadata() -> Dict[str, Any]:
    """Return default metadata when extraction fails"""
    return {
        "location_tag": "Not Provided",
        "condition": "Not Provided",
        "action": "Not Provided",
        "priority": "Not Provided",
        "description": "Not Provided",
        "value": None,
        "units": None,
    }


def _extract_metadata_from_csv(file_path: str, source: str, bin_start: str, bin_end: str) -> Dict[str, Any]:
    """
    Extracts metadata by reading the CSV file directly and filtering by source and time range.
    Uses the same CSV reading logic as other functions in the codebase.
    """
    try:
        logger.info(f"Extracting metadata for source {source} from {file_path} between {bin_start} and {bin_end}")
        
        # Use the same CSV reading approach as read_pvc_file
        expected_columns = [
            "Event Time", "Location Tag", "Source", "Condition", 
            "Action", "Priority", "Description", "Value", "Units"
        ]
        
        # Try reading with comma separator first
        try:
            df = pd.read_csv(
                file_path, 
                sep=',',
                encoding='utf-8', 
                engine='python',
                skipinitialspace=True,
                quotechar='"',
                on_bad_lines='skip',
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                skiprows=8  # Skip first 8 rows which contain metadata
            )
        except Exception:
            # Fallback to tab separator
            df = pd.read_csv(
                file_path, 
                sep='\t',
                encoding='utf-8', 
                engine='python',
                skipinitialspace=True,
                quotechar='"',
                on_bad_lines='skip',
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                skiprows=8
            )
        
        logger.info(f"Read CSV with columns: {list(df.columns)}")
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # Ensure we have all required columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
        
        # Select only the columns we want
        df = df[expected_columns]
        
        # Replace any remaining whitespace-only cells with empty string
        df = df.replace(r'^\s*$', "", regex=True)
        
        logger.info(f"Total rows before source filter: {len(df)}")
        
        # Filter rows by source
        df = df[df["Source"].str.strip() == source]
        logger.info(f"Rows after source filter for {source}: {len(df)}")
        
        if df.empty:
            logger.warning(f"No records found for source {source}")
            return _default_metadata()

        # Convert Event Time to datetime - handle multiple formats
        df["Event Time"] = pd.to_datetime(df["Event Time"], errors="coerce", infer_datetime_format=True)
        df = df.dropna(subset=["Event Time"])
        
        if df.empty:
            logger.warning(f"No valid timestamps found for source {source}")
            return _default_metadata()

        logger.info(f"Rows after datetime conversion: {len(df)}")
        logger.info(f"Sample timestamps: {df['Event Time'].head().tolist()}")

        # Convert bin times to datetime with timezone handling
        # The bin times are in UTC, but CSV times might be in local timezone
        bin_start_dt = pd.to_datetime(bin_start).tz_convert(None) if pd.to_datetime(bin_start).tz else pd.to_datetime(bin_start)
        bin_end_dt = pd.to_datetime(bin_end).tz_convert(None) if pd.to_datetime(bin_end).tz else pd.to_datetime(bin_end)
        
        # Make CSV timestamps timezone-naive for comparison
        df["Event Time"] = df["Event Time"].dt.tz_localize(None) if df["Event Time"].dt.tz else df["Event Time"]
        
        logger.info(f"Filtering between {bin_start_dt} and {bin_end_dt}")
        
        # Use a wider time window to account for timezone differences
        # Expand the search window by 12 hours on each side
        expanded_start = bin_start_dt - pd.Timedelta(hours=12)
        expanded_end = bin_end_dt + pd.Timedelta(hours=12)
        
        mask = (df["Event Time"] >= expanded_start) & (df["Event Time"] <= expanded_end)
        df_filtered = df[mask]
        
        logger.info(f"Rows after time filter (expanded window): {len(df_filtered)}")

        # If no records in expanded window, try the original window
        if df_filtered.empty:
            mask = (df["Event Time"] >= bin_start_dt) & (df["Event Time"] <= bin_end_dt)
            df_filtered = df[mask]
            logger.info(f"Rows after time filter (exact window): {len(df_filtered)}")

        # If still no records, just take the first record for this source
        if df_filtered.empty:
            logger.warning(f"No records in time window, using first available record for source {source}")
            df_filtered = df.head(1)

        if df_filtered.empty:
            logger.warning(f"No records found at all for source {source}")
            return _default_metadata()

        # Get the first row for metadata
        first_row = df_filtered.iloc[0]
        
        result = {
            "location_tag": str(first_row.get("Location Tag", "")).strip() or "Not Provided",
            "condition": str(first_row.get("Condition", "")).strip() or "Not Provided",
            "action": str(first_row.get("Action", "")).strip() or "Not Provided",
            "priority": str(first_row.get("Priority", "")).strip() or "Not Provided",
            "description": str(first_row.get("Description", "")).strip() or "Not Provided",
            "value": first_row.get("Value", None) if str(first_row.get("Value", "")).strip() else None,
            "units": first_row.get("Units", None) if str(first_row.get("Units", "")).strip() else None,
        }
        
        logger.info(f"Extracted metadata: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Metadata extraction failed for {file_path}: {e}")
        return _default_metadata()

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
        bin_size: str = '10min',
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


import pandas as pd
import math
from typing import Dict, Any, List

def group_events_by_source_with_timegap(df, bin_size: str = '10min', max_gap_minutes: int = None):
    """
    Groups events by source into FIXED 10-min bins (non-overlapping).
    - Bins via dt.floor(bin_size) for strict "in 10 minutes" checks.
    - Ignores gaps; counts hits per bin.
    - If max_gap_minutes provided, optionally split bins with large internal gaps (future-proof).
    - Returns DF with hits, first/last ts, fixed window=10 min per bin.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("__ts__").reset_index(drop=True)
    df['bin_start'] = df['__ts__'].dt.floor(bin_size)
    df['bin_end'] = df['bin_start'] + pd.Timedelta(bin_size)

    # Group by source + bin, agg stats
    def bin_stats(group):
        ts = group['__ts__']
        return pd.Series({
            "__source__": group.name[0],  # source
            "bin_start": group.name[1],   # bin_start as index
            "hits": len(group),
            "first_event_ts": ts.min(),
            "last_event_ts": ts.max(),
            "duration_seconds": (ts.max() - ts.min()).total_seconds() if len(ts) > 1 else 0,
            "window_minutes": int(pd.Timedelta(bin_size).total_seconds() / 60),  # Fixed 10
        })

    grouped_runs = df.groupby(['__source__', 'bin_start']).apply(bin_stats).reset_index(drop=True)

    # Compute rate (hits / 10 min, even if actual duration <10)
    grouped_runs['window_minutes'] = 10  # Override if needed
    grouped_runs['rate_per_min'] = round(grouped_runs['hits'] / 10.0, 6)

    # Only include non-empty bins (hits >=1)
    grouped_runs = grouped_runs[grouped_runs['hits'] >= 1]

    return grouped_runs


def detect_flood_events_for_source(
    src_df: pd.DataFrame,
    threshold: int = 10,
    time_window_minutes: int = 10,
) -> List[Dict[str, Any]]:
    """
    Detect dynamic sliding-window flood events for a single source using a deque window.
    - src_df must contain a timezone-aware timestamp column '__ts__' and be sorted by time.
    - Returns merged flood clusters with accurate duration and rate.
    """
    from collections import deque

    if src_df.empty:
        return []

    # Ensure sorted
    s = src_df.sort_values("__ts__").reset_index(drop=True)

    window = deque()
    candidates: List[Dict[str, Any]] = []
    max_span = pd.Timedelta(minutes=time_window_minutes)

    for ts in s["__ts__"].tolist():
        window.append(ts)
        while window and (window[-1] - window[0]) > max_span:
            window.popleft()
        if len(window) >= int(threshold):
            candidates.append({
                "start": window[0],
                "end": window[-1],
                "count": len(window),
                # Track exact 10-min window
                "peak_start": window[0],
                "peak_end": window[-1],
            })

    if not candidates:
        return []

    # Merge overlapping/adjacent candidates to maximal clusters
    epsilon = pd.Timedelta(seconds=1)
    merged: List[Dict[str, Any]] = []
    current = dict(candidates[0])
    current.setdefault("peak_start", current["start"]) 
    current.setdefault("peak_end", current["end"]) 
    for nxt in candidates[1:]:
        if nxt["start"] <= current["end"] + epsilon:
            current["end"] = max(current["end"], nxt["end"])
            # If next has a higher count, adopt its peak window
            if int(nxt["count"]) > int(current.get("count", 0)):
                current["count"] = int(nxt["count"])
                current["peak_start"] = nxt.get("peak_start", nxt["start"]) 
                current["peak_end"] = nxt.get("peak_end", nxt["end"]) 
        else:
            merged.append(current)
            current = dict(nxt)
            current.setdefault("peak_start", current["start"]) 
            current.setdefault("peak_end", current["end"]) 
    merged.append(current)

    # Compute duration and rate for each merged cluster
    events: List[Dict[str, Any]] = []
    for m in merged:
        duration_min = (m["end"] - m["start"]).total_seconds() / 60.0
        rate_per_min = (m.get("count", 0) / duration_min) if duration_min > 0 else 0.0
        events.append({
            "start": m["start"],
            "end": m["end"],
            "count": int(m.get("count", 0)),
            "duration_min": float(round(duration_min, 6)),
            "rate_per_min": float(round(rate_per_min, 6)),
            "peak_start": m.get("peak_start", m["start"]),
            "peak_end": m.get("peak_end", m["start"] + pd.Timedelta(minutes=time_window_minutes)),
        })

    return events


def compute_pvcI_file_health(
    file_path: str,
    config: HealthConfig = DEFAULT_CONFIG,
    include_details: bool = True
) -> Dict[str, Any]:
    """Compute health for one PVC-I day file using FIXED bin grouping per source."""
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
    if df.empty:
        return {
            "filename": filename,
            "num_sources": 0,
            "health_pct": 100.0,
            "unhealthy_count": 0,
            "source_details": {}
        }

    # --- Normalize & sort ---
    df["__ts__"] = pd.to_datetime(df["Event Time"], errors="coerce", utc=True)
    df["__source__"] = df["Source"].astype(str).str.strip()
    df = df.dropna(subset=["__ts__", "__source__"]).sort_values("__ts__").reset_index(drop=True)
    if df.empty:
        return {
            "filename": filename,
            "num_sources": 0,
            "health_pct": 100.0,
            "unhealthy_count": 0,
            "source_details": {}
        }

    # --- FIXED bin grouping for rollups ---
    grouped_runs = group_events_by_source_with_timegap(
        df, bin_size=config.bin_size
    )

    alarm_threshold = int(getattr(config, "alarm_threshold", 10) or 10)
    bin_minutes = int(pd.Timedelta(config.bin_size).total_seconds() // 60)

    # --- Dynamic sliding-window floods per source ---
    source_events: Dict[str, List[Dict[str, Any]]] = {}
    for src, s_df in df.groupby("__source__"):
        source_events[src] = detect_flood_events_for_source(
            s_df[["__ts__"]], threshold=alarm_threshold, time_window_minutes=bin_minutes
        )

    # Initialize unhealthy flags from fixed threshold, then OR with dynamic overlaps
    grouped_runs["is_unhealthy"] = grouped_runs["hits"] >= alarm_threshold
    if not grouped_runs.empty:
        for src, src_runs in grouped_runs.groupby("__source__"):
            events = source_events.get(src) or []
            if not events:
                continue
            idx = src_runs.index
            bin_start = src_runs["bin_start"]
            bin_end = bin_start + pd.Timedelta(minutes=bin_minutes)
            overlapped = pd.Series(False, index=idx)
            for ev in events:
                ev_start = ev["start"]
                ev_end = ev["end"]
                mask = (bin_start < ev_end) & (bin_end > ev_start)
                overlapped = overlapped | mask
            grouped_runs.loc[idx, "is_unhealthy"] = grouped_runs.loc[idx, "is_unhealthy"] | overlapped

    # --- Build source_details ---
    source_details: Dict[str, Any] = {}
    unhealthy_total = 0
    for src, src_runs in grouped_runs.groupby("__source__"):
        # Work on updated per-source view
        total_runs = len(src_runs)
        unhealthy_runs = int(src_runs["is_unhealthy"].sum())
        healthy_runs = total_runs - unhealthy_runs
        unhealthy_total += unhealthy_runs

        health_pct = round((healthy_runs / float(total_runs)) * 100.0, 6) if total_runs > 0 else 100.0
        details: List[Dict[str, Any]] = []

        if include_details and unhealthy_runs > 0:
            # Sort unhealthy by hits desc, then bin_start asc
            for _, row in src_runs.loc[src_runs["is_unhealthy"]].sort_values(
                ["hits", "bin_start"], ascending=[False, True]
            ).iterrows():
                hits = int(row["hits"])
                over_by = hits - alarm_threshold
                over_pct = round((over_by / alarm_threshold) * 100.0, 6) if alarm_threshold > 0 else None

                # Attach best overlapping dynamic event info, if any
                bin_start_ts = row["bin_start"]
                bin_end_ts = bin_start_ts + pd.Timedelta(minutes=bin_minutes)
                best_ev = None
                for ev in (source_events.get(src) or []):
                    if (bin_start_ts < ev["end"]) and (bin_end_ts > ev["start"]):
                        if (best_ev is None) or (ev["count"] > best_ev["count"]) or (
                            ev["count"] == best_ev["count"] and ev["rate_per_min"] > best_ev["rate_per_min"]
                        ):
                            best_ev = ev

                detail = {
                    "source": str(src),
                    "status": "unhealthy",
                    "bin_start": bin_start_ts.isoformat(),
                    "bin_end": bin_end_ts.isoformat(),  # Fixed end
                    "window_minutes": int(row["window_minutes"]),
                    "hits": hits,
                    "threshold": alarm_threshold,
                    "over_by": over_by,
                    "over_pct": over_pct,
                    "rate_per_min": row["rate_per_min"],
                }

                if best_ev is not None:
                    detail.update({
                        "duration_min": round(float(best_ev["duration_min"]), 6),
                        "flood_count": int(best_ev["count"]),
                        "event_rate_per_min": round(float(best_ev["rate_per_min"]), 6),
                        # New: Exact 10-minute peak window producing flood_count
                        "peak_window_start": best_ev.get("peak_start").isoformat() if best_ev.get("peak_start") is not None else None,
                        "peak_window_end": best_ev.get("peak_end").isoformat() if best_ev.get("peak_end") is not None else None,
                        "peak_rate_per_min": round((int(best_ev["count"]) / 10.0), 6),
                    })

                # Enrich with static CSV metadata using per-file index (fast path)
                try:
                    metadata = _get_metadata_fast(file_path, str(src))
                    detail.update({
                        "location_tag": metadata.get("location_tag"),
                        "condition": metadata.get("condition"),
                        "action": metadata.get("action"),
                        "priority": metadata.get("priority"),
                        "description": metadata.get("description"),
                        "setpoint_value": metadata.get("value"),
                        "units": metadata.get("units"),
                    })
                except Exception:
                    pass

                details.append(detail)

        source_details[str(src)] = {
            "total_bins": int(total_runs),  # Non-empty 10-min bins
            "bins_with_events": int(total_runs),
            "healthy_bins": int(healthy_runs),
            "unhealthy_bins": int(unhealthy_runs),
            "health_pct": health_pct,
            "unhealthy_bin_details": details
        }

    num_sources = grouped_runs["__source__"].nunique()
    overall_health = sum(d["health_pct"] for d in source_details.values()) / float(num_sources) if num_sources > 0 else 100.0

    return {
        "filename": filename,
        "num_sources": int(num_sources),
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
                if per_file_timeout is None:
                    # No global timeout: collect all results as they finish
                    for fut in as_completed(futures):
                        try:
                            file_results.append(fut.result())
                        except Exception as e:
                            errors.append(str(e))
                else:
                    # Maintain behavior with a global wait, but only skip those not done
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

    # Weighted overall using run totals (not time bins)
    total_bins_all = 0
    total_healthy_all = 0
    for r in daily_results:
        source_details = r.get('source_details') or {}
        total_bins_file = 0
        healthy_bins_file = 0
        for sd in source_details.values():
            tb = int(sd.get('total_bins', 0) or 0)
            hb = int(sd.get('healthy_bins', 0) or 0)
            total_bins_file += tb
            healthy_bins_file += hb
        total_bins_all += total_bins_file
        total_healthy_all += healthy_bins_file

    overall_health_weighted = (total_healthy_all / total_bins_all * 100) if total_bins_all > 0 else 100

    # Aggregate per-source health across files from source_details
    all_source_health: Dict[str, List[float]] = {}
    for result in daily_results:
        for src, sd in (result.get('source_details') or {}).items():
            hp = float(sd.get('health_pct', 100.0) or 100.0)
            all_source_health.setdefault(src, []).append(hp)

    avg_source_health = { src: (sum(vals) / len(vals)) for src, vals in all_source_health.items() }

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
            # Prefer dynamic event window if available, else fall back to bin window
            meta_start = bin_detail.get("event_start", bin_detail["bin_start"])  # ISO 8601
            meta_end = bin_detail.get("event_end", bin_detail["bin_end"])      # ISO 8601
            metadata = _extract_metadata_from_csv(
                file_path, source, meta_start, meta_end
            )

            unhealthy_records.append({
                "event_time": bin_detail["bin_start"],
                "bin_end": bin_detail["bin_end"],
                "source": source,
                "hits": bin_detail["hits"],
                "threshold": bin_detail["threshold"],
                "over_by": bin_detail["over_by"],
                "rate_per_min": bin_detail["rate_per_min"],
                "event_start": bin_detail.get("event_start"),
                "event_end": bin_detail.get("event_end"),
                "duration_min": bin_detail.get("duration_min"),
                "flood_count": bin_detail.get("flood_count"),
                "event_rate_per_min": bin_detail.get("event_rate_per_min"),
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


# Per-file metadata index to avoid re-reading CSVs for every unhealthy bin
@lru_cache(maxsize=128)
def _build_metadata_index(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Build a mapping of Source -> static metadata from one CSV read."""
    try:
        expected_columns = [
            "Event Time", "Location Tag", "Source", "Condition",
            "Action", "Priority", "Description", "Value", "Units"
        ]

        # Prefer comma, fallback to tab
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
                skiprows=8
            )
        except Exception:
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

        df.columns = df.columns.str.strip()
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""

        df = df[expected_columns]
        df = df.replace(r'^\s*$', "", regex=True)

        first_rows = df.drop_duplicates(subset=["Source"], keep="first")
        index: Dict[str, Dict[str, Any]] = {}
        for _, row in first_rows.iterrows():
            src = str(row.get("Source", "")).strip()
            if not src:
                continue
            index[src] = {
                "location_tag": str(row.get("Location Tag", "")).strip() or "Not Provided",
                "condition": str(row.get("Condition", "")).strip() or "Not Provided",
                "action": str(row.get("Action", "")).strip() or "Not Provided",
                "priority": str(row.get("Priority", "")).strip() or "Not Provided",
                "description": str(row.get("Description", "")).strip() or "Not Provided",
                "value": (row.get("Value", None) if str(row.get("Value", "")).strip() else None),
                "units": (row.get("Units", None) if str(row.get("Units", "")).strip() else None),
            }

        return index
    except Exception:
        return {}


def _get_metadata_fast(file_path: str, source: str) -> Dict[str, Any]:
    index = _build_metadata_index(file_path)
    return index.get(source, _default_metadata())
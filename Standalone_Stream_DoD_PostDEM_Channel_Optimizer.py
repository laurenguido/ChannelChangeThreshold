# -*- coding: utf-8 -*-
"""Standalone stream-network + post-event DEM + DoD channel envelope optimizer.

A simplified adaptation of 01_PostDEM_Channel_Polygon_Optimizer.py.
Uses ONE user-supplied stream network and no CO2CHAIN, watershed IDs,
pre-event DEM, mapped flow points, or uncertainty rasters.

HOW TO RUN: Edit the three input paths in USER SETTINGS, then run in Spyder,
VS Code, or a Python terminal. Inputs must overlap spatially and the DEM must
have a projected horizontal CRS in metres; DEM vertical units must be metres.
Stream network must contain LineString/MultiLineString geometries.

DEPENDENCIES: pip install numpy pandas scipy rasterio geopandas shapely matplotlib

IMPORTANT METHOD NOTES
* ROUTED_D8 calculates a simple steepest-descent, unconditioned DEM D8 surface
  routing to supplied stream cells, then HAND = DEM elevation - reached stream
  cell elevation. Local sinks/unconnected cells remain NoData, not extrapolated.
  This is NOT identical to the original CO2CHAIN-conditioned HAND calculation.
* NEAREST_STREAM is a simpler proximity reference (NOT hydrologically routed).
* With only a DoD, abs(DoD) > MIN_ABS_CHANGE_M is a magnitude proxy, NOT a
  statistically significant change claim. For deposition-only, set
  CHANGE_MODE='deposition'. No LoD95 is calculated without uncertainty.
* Cutoff search preserves the original envelope morphology, marginal qualified
  share, BIC-based changepoint selector, and no-changepoint fallback.
"""
from __future__ import annotations
import json
import math
import traceback
from pathlib import Path
from typing import Any
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject
from scipy import ndimage
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

# ============================================================================
# USER SETTINGS: EDIT THESE PATHS
# ============================================================================
STREAM_NETWORK = Path(r"C:\GIS\streams.gpkg")  # .shp, .gpkg, etc.
STREAM_LAYER = None                         # set if a GPKG has multiple layers
POST_DEM = Path(r"C:\GIS\post_dem.tif")
DOD = Path(r"C:\GIS\dod.tif")
OUTPUT_DIR = Path(r"C:\GIS\Channel_Optimizer_Output")

HAND_METHOD = "ROUTED_D8"                 # ROUTED_D8 or NEAREST_STREAM
MIN_ABS_CHANGE_M = 0.10                 # fixed magnitude threshold, not LoD95
CHANGE_MODE = "either"                 # either, deposition, erosion
# DoD convention assumed: post minus pre (positive = deposition).
MAX_ABS_DOD_CHANGE_M = 20.0            # use None to disable extreme exclusion
# For a project whose DoD uses the opposite sign, invert CHANGE_MODE meaning.

HAND_CUTOFF_MIN_M = 0.2
HAND_CUTOFF_MAX_M = 5.0
HAND_CUTOFF_STEP_M = 0.1
CHANGEPOINT_MIN_SEGMENT_POINTS = 3
CHANGEPOINT_MEDIAN_WINDOW = 3
CHANGEPOINT_FALLBACK = "best_marginal_efficiency"
SELECTION_METHOD = "marginal_changepoint_bic"
MAX_COVERAGE_FAILURE_PCT = None
COMPUTE_PARETO_DIAGNOSTIC = True

VECTOR_MIN_PATCH_AREA_M2 = 4.0
VECTOR_MAX_HOLE_AREA_M2 = 4.0
VECTOR_RASTER_CLOSE_RADIUS_M = 2.0
VECTOR_RASTER_EXPAND_M = 1.0
VECTOR_SMOOTH_BUFFER_M = 0.75
VECTOR_SIMPLIFY_TOLERANCE_M = 0.5
VECTOR_BUFFER_QUAD_SEGS = 8
VECTOR_ENVELOPE_CLIP_TO_VALID_HAND = True

# Stream lines buffer only for rasterization sub-pixel intersections. One raster
# cell at most if set to 0 (all_touched=True). Set a positive value if desired.
STREAM_BUFFER_M = 0.0
WRITE_CANDIDATE_GPKG = False             # may be large for fine cutoff steps
WRITE_QC_FIGURE = True
OVERWRITE_OUTPUTS = True

# ============================================================================
# Original optimizer's morphology and BIC selector (adapted without CO2CHAIN)
# ============================================================================
def _remove_small_mask_patches(mask: np.ndarray, cell_area_m2: float, min_area_m2: float) -> np.ndarray:
    """Remove isolated True components smaller than the requested map area."""
    if min_area_m2 is None or float(min_area_m2) <= 0:
        return mask.astype(bool, copy=True)
    out = mask.astype(bool, copy=True)
    labels, n = ndimage.label(out, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return out
    counts = np.bincount(labels.ravel())
    min_cells = max(1, int(math.ceil(float(min_area_m2) / max(float(cell_area_m2), 1e-12))))
    remove_ids = np.where(counts < min_cells)[0]
    remove_ids = remove_ids[remove_ids != 0]
    if remove_ids.size:
        out[np.isin(labels, remove_ids)] = False
    return out

def _fill_small_mask_holes(mask: np.ndarray, valid: np.ndarray, cell_area_m2: float, max_hole_area_m2: float) -> np.ndarray:
    """Fill small enclosed False components within a channel mask, respecting the valid HAND domain."""
    if max_hole_area_m2 is None or float(max_hole_area_m2) <= 0:
        return mask.astype(bool, copy=True)
    out = mask.astype(bool, copy=True)
    holes = valid & ~out
    labels, n = ndimage.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return out
    valid_edge = valid & ~ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)
    exterior_ids = set(np.unique(labels[valid_edge & (labels > 0)]).tolist())
    counts = np.bincount(labels.ravel())
    max_cells = int(math.floor(float(max_hole_area_m2) / max(float(cell_area_m2), 1e-12)))
    if max_cells < 1:
        return out
    for label_id in range(1, n + 1):
        if label_id in exterior_ids:
            continue
        if counts[label_id] <= max_cells:
            out[labels == label_id] = True
    return out

def _metric_disk_structure(transform, radius_m: float) -> np.ndarray:
    """Return an approximately circular binary morphology kernel in projected metres."""
    radius_m = float(radius_m or 0.0)
    if radius_m <= 0:
        return np.ones((1, 1), dtype=bool)
    xres = math.hypot(transform.a, transform.d)
    yres = math.hypot(transform.b, transform.e)
    if xres <= 0 or yres <= 0:
        raise RuntimeError('Invalid raster resolution for vector-envelope morphology.')
    rx = max(1, int(math.ceil(radius_m / xres)))
    ry = max(1, int(math.ceil(radius_m / yres)))
    yy, xx = np.mgrid[-ry:ry+1, -rx:rx+1]
    dist2 = (xx * xres) ** 2 + (yy * yres) ** 2
    return dist2 <= (radius_m + 1e-9) ** 2

def _regularize_channel_mask(mask: np.ndarray, valid: np.ndarray, transform, cell_area_m2: float) -> tuple[np.ndarray, dict]:
    """
    Build a generalized channel envelope in the raster domain before polygonization.

    Sequence:
      1) remove tiny isolated patches
      2) fill tiny enclosed holes
      3) circular morphological closing to bridge notches/gaps
      4) circular outward dilation/expansion
      5) fill newly enclosed small holes again

    This is intentionally a mapping/analysis envelope, not a replacement for the
    authoritative HAND threshold raster.
    """
    raw = mask.astype(bool, copy=True)
    work = _remove_small_mask_patches(raw, cell_area_m2, VECTOR_MIN_PATCH_AREA_M2)
    work = _fill_small_mask_holes(work, valid, cell_area_m2, VECTOR_MAX_HOLE_AREA_M2)

    before_close = int(work.sum())
    close_r = float(VECTOR_RASTER_CLOSE_RADIUS_M or 0.0)
    if close_r > 0:
        st = _metric_disk_structure(transform, close_r)
        work = ndimage.binary_closing(work, structure=st, border_value=0)
        if VECTOR_ENVELOPE_CLIP_TO_VALID_HAND:
            work &= valid
    after_close = int(work.sum())

    before_expand = int(work.sum())
    expand_r = float(VECTOR_RASTER_EXPAND_M or 0.0)
    if expand_r > 0:
        st = _metric_disk_structure(transform, expand_r)
        work = ndimage.binary_dilation(work, structure=st, border_value=0)
        if VECTOR_ENVELOPE_CLIP_TO_VALID_HAND:
            work &= valid
    after_expand = int(work.sum())

    work = _fill_small_mask_holes(work, valid, cell_area_m2, VECTOR_MAX_HOLE_AREA_M2)
    work = _remove_small_mask_patches(work, cell_area_m2, VECTOR_MIN_PATCH_AREA_M2)
    if VECTOR_ENVELOPE_CLIP_TO_VALID_HAND:
        work &= valid

    info = {
        'raw_cells': int(raw.sum()),
        'cells_after_initial_cleanup': int(before_close),
        'cells_added_by_closing_net': int(after_close - before_close),
        'cells_added_by_expansion_net': int(after_expand - before_expand),
        'envelope_cells_final': int(work.sum()),
    }
    return work.astype(bool), info

def _polygonize_mask(mask: np.ndarray, transform) -> Any:
    """Polygonize a boolean mask and dissolve all pieces into one geometry."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    geoms = [
        shapely_shape(geom)
        for geom, value in shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform,
            connectivity=8,
        )
        if int(value) == 1
    ]
    if not geoms:
        return None
    geom = unary_union(geoms)
    if geom is not None and not geom.is_empty and not geom.is_valid:
        geom = geom.buffer(0)
    return None if geom is None or geom.is_empty else geom

def _smooth_channel_geometry(geom, smooth_m: float, simplify_m: float):
    """Round residual corners and lightly simplify after raster-domain envelope construction."""
    if geom is None or geom.is_empty:
        return None
    out = geom
    smooth_m = float(smooth_m or 0.0)
    simplify_m = float(simplify_m or 0.0)
    if smooth_m > 0:
        out = out.buffer(
            smooth_m,
            resolution=int(VECTOR_BUFFER_QUAD_SEGS),
            join_style='round',
        ).buffer(
            -smooth_m,
            resolution=int(VECTOR_BUFFER_QUAD_SEGS),
            join_style='round',
        )
    if simplify_m > 0 and out is not None and not out.is_empty:
        out = out.simplify(simplify_m, preserve_topology=True)
    if out is not None and not out.is_empty and not out.is_valid:
        out = out.buffer(0)
    return None if out is None or out.is_empty else out

def cutoff_values() -> list[float]:
    lo = float(HAND_CUTOFF_MIN_M)
    hi = float(HAND_CUTOFF_MAX_M)
    step = float(HAND_CUTOFF_STEP_M)
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(step) and step > 0 and hi >= lo):
        raise ValueError("Invalid HAND cutoff range settings")
    n = int(round((hi - lo) / step))
    vals = [round(lo + i * step, 10) for i in range(n + 1)]
    if vals[-1] < hi - 1e-9:
        vals.append(hi)
    return vals

def _rasterize_final_geom(geom, profile: dict) -> np.ndarray:
    if geom is None or geom.is_empty:
        return np.zeros((profile["height"], profile["width"]), dtype=bool)
    return rasterize(
        [(geom, 1)], out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"], fill=0, default_value=1,
        dtype="uint8", all_touched=False,
    ).astype(bool)

def _candidate_for_cutoff(hand: np.ndarray, hand_valid: np.ndarray, profile: dict, cutoff: float) -> dict:
    transform = profile["transform"]
    cell_area = abs(float(transform.a * transform.e - transform.b * transform.d))
    raw = hand_valid & np.isfinite(hand) & (hand <= float(cutoff))
    envelope_mask, morph = _regularize_channel_mask(raw, hand_valid, transform, cell_area)
    raw_geom = _polygonize_mask(raw, transform)
    env_geom = _polygonize_mask(envelope_mask, transform)
    env_geom = _smooth_channel_geometry(env_geom, VECTOR_SMOOTH_BUFFER_M, VECTOR_SIMPLIFY_TOLERANCE_M)
    eval_mask = _rasterize_final_geom(env_geom, profile)
    return {
        "cutoff_m": float(cutoff), "raw_mask": raw, "envelope_mask": envelope_mask,
        "raw_geom": raw_geom, "envelope_geom": env_geom, "eval_mask": eval_mask,
        "morphology": morph, "cell_area_m2": cell_area,
    }

def _quality_metrics(candidate: dict, quality: dict) -> dict:
    mask = candidate["eval_mask"]
    n = int(mask.sum())
    cell_area = float(candidate["cell_area_m2"])
    area = float(candidate["envelope_geom"].area) if candidate["envelope_geom"] is not None else 0.0

    def count(name):
        return int(np.count_nonzero(mask & quality[name]))

    missing = count("missing")
    below = count("below_lod")
    extreme = count("extreme")
    qual = count("qualified")
    disq = count("disqualified")
    # Missing/NoData and extreme-QC cells are true coverage/rejection failures.
    # Below-LoD cells remain valid observations, but no change is detectable above
    # the local uncertainty threshold. Keep these concepts separate.
    coverage_failure = missing + extreme
    observed_valid = below + qual
    denom = max(n, 1)
    eval_area = float(n * cell_area)
    qualified_area = float(qual * cell_area)
    disqualified_area = float(disq * cell_area)

    return {
        "cutoff_m": candidate["cutoff_m"],
        "polygon_cells": n,
        "polygon_vector_area_m2": area,
        "polygon_eval_area_m2": eval_area,
        "missing_cells": missing,
        "below_lod_cells": below,
        "extreme_change_cells": extreme,
        "qualified_change_cells": qual,
        "disqualified_cells": disq,
        "coverage_failure_cells": int(coverage_failure),
        "observed_valid_cells": int(observed_valid),
        "missing_area_m2": float(missing * cell_area),
        "below_lod_area_m2": float(below * cell_area),
        "extreme_change_area_m2": float(extreme * cell_area),
        "qualified_area_m2": qualified_area,
        "disqualified_area_m2": disqualified_area,
        "coverage_failure_area_m2": float(coverage_failure * cell_area),
        "observed_valid_area_m2": float(observed_valid * cell_area),
        "missing_pct": 100.0 * missing / denom,
        "below_lod_pct": 100.0 * below / denom,
        "extreme_change_pct": 100.0 * extreme / denom,
        "qualified_change_pct": 100.0 * qual / denom,
        "disqualified_pct": 100.0 * disq / denom,
        "coverage_failure_pct": 100.0 * coverage_failure / denom,
        "observed_valid_pct": 100.0 * observed_valid / denom,
        **{f"morph_{k}": v for k, v in candidate["morphology"].items()},
    }

def _add_incremental_metrics(metrics: list[dict]) -> list[dict]:
    """Add marginal area-gain diagnostics in ascending HAND-cutoff order."""
    rows = sorted((dict(m) for m in metrics), key=lambda m: float(m["cutoff_m"]))
    previous = None
    for row in rows:
        if previous is None:
            row.update({
                "delta_polygon_area_m2": np.nan,
                "delta_qualified_area_m2": np.nan,
                "delta_disqualified_area_m2": np.nan,
                "delta_coverage_failure_area_m2": np.nan,
                "delta_below_lod_area_m2": np.nan,
                "delta_disqualified_pct_points": np.nan,
                "marginal_bad_per_good_area": np.nan,
                "incremental_qualified_share_pct": np.nan,
                "incremental_below_lod_share_pct": np.nan,
                "incremental_coverage_failure_share_pct": np.nan,
            })
        else:
            d_total = float(row["polygon_eval_area_m2"] - previous["polygon_eval_area_m2"])
            d_good = float(row["qualified_area_m2"] - previous["qualified_area_m2"])
            d_bad = float(row["disqualified_area_m2"] - previous["disqualified_area_m2"])
            d_fail = float(row["coverage_failure_area_m2"] - previous["coverage_failure_area_m2"])
            d_below = float(row["below_lod_area_m2"] - previous["below_lod_area_m2"])
            d_bad_pct = float(row["disqualified_pct"] - previous["disqualified_pct"])
            if d_good > 1e-12:
                bad_per_good = d_bad / d_good
            elif d_bad > 1e-12:
                bad_per_good = np.inf
            else:
                bad_per_good = np.nan
            inc_good_share = 100.0 * d_good / d_total if d_total > 1e-12 else np.nan
            inc_below_share = 100.0 * d_below / d_total if d_total > 1e-12 else np.nan
            inc_fail_share = 100.0 * d_fail / d_total if d_total > 1e-12 else np.nan
            row.update({
                "delta_polygon_area_m2": d_total,
                "delta_qualified_area_m2": d_good,
                "delta_disqualified_area_m2": d_bad,
                "delta_coverage_failure_area_m2": d_fail,
                "delta_below_lod_area_m2": d_below,
                "delta_disqualified_pct_points": d_bad_pct,
                "marginal_bad_per_good_area": float(bad_per_good) if np.isfinite(bad_per_good) else bad_per_good,
                "incremental_qualified_share_pct": float(inc_good_share) if np.isfinite(inc_good_share) else np.nan,
                "incremental_below_lod_share_pct": float(inc_below_share) if np.isfinite(inc_below_share) else np.nan,
                "incremental_coverage_failure_share_pct": float(inc_fail_share) if np.isfinite(inc_fail_share) else np.nan,
            })
        previous = row
    return rows

def _rolling_median_finite(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median used only to stabilize changepoint detection."""
    values = np.asarray(values, dtype=float)
    if int(window) <= 1:
        return values.copy()
    if int(window) % 2 == 0:
        raise ValueError("CHANGEPOINT_MEDIAN_WINDOW must be odd (or 1 to disable)")
    return pd.Series(values).rolling(int(window), center=True, min_periods=1).median().to_numpy(dtype=float)

def _linear_fit_sse(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """OLS line fit returning SSE, intercept, slope, and fitted values."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.inf, np.nan, np.nan, np.full_like(y, np.nan)
    if np.ptp(x) <= 1e-12:
        intercept = float(np.mean(y))
        fitted = np.full_like(y, intercept)
        return float(np.sum((y - fitted) ** 2)), intercept, 0.0, fitted
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    sse = float(np.sum((y - fitted) ** 2))
    return sse, float(intercept), float(slope), fitted

def _bic_from_sse(sse: float, n: int, k: int) -> float:
    """Gaussian least-squares BIC with a small numerical floor on residual variance."""
    if n <= 0 or not np.isfinite(sse):
        return np.inf
    mse = max(float(sse) / float(n), 1e-12)
    return float(n * math.log(mse) + k * math.log(max(n, 1)))

def _annotate_pareto_diagnostic(metrics: list[dict]) -> tuple[list[dict], dict]:
    """Annotate Pareto membership and geometric-knee diagnostics without selecting by them."""
    rows = [dict(m) for m in metrics]
    usable = [m for m in rows if int(m.get("polygon_cells", 0)) > 0
              and np.isfinite(m.get("qualified_area_m2", np.nan))
              and np.isfinite(m.get("disqualified_pct", np.nan))]
    frontier = []
    if COMPUTE_PARETO_DIAGNOSTIC:
        for a in usable:
            dominated = False
            ag, ab = float(a["qualified_area_m2"]), float(a["disqualified_pct"])
            for b in usable:
                if b is a:
                    continue
                bg, bb = float(b["qualified_area_m2"]), float(b["disqualified_pct"])
                if (bg >= ag - 1e-12 and bb <= ab + 1e-12 and
                        (bg > ag + 1e-12 or bb < ab - 1e-12)):
                    dominated = True
                    break
            if not dominated:
                frontier.append(dict(a))
        frontier.sort(key=lambda m: (float(m["qualified_area_m2"]), float(m["disqualified_pct"])))
    fkeys = {round(float(m["cutoff_m"]), 10) for m in frontier}
    for row in rows:
        row["pareto_optimal"] = round(float(row["cutoff_m"]), 10) in fkeys
        row["pareto_knee_score"] = np.nan
        row["pareto_x_normalized"] = np.nan
        row["pareto_y_normalized"] = np.nan

    diag = {
        "pareto_frontier_cutoffs_m": [float(m["cutoff_m"]) for m in frontier],
        "pareto_frontier_size": int(len(frontier)),
        "pareto_knee_cutoff_m": None,
        "pareto_knee_prominence": 0.0,
    }
    if len(frontier) >= 3:
        x = np.asarray([float(m["qualified_area_m2"]) for m in frontier], dtype=float)
        y = np.asarray([float(m["disqualified_pct"]) for m in frontier], dtype=float)
        xr, yr = float(np.ptp(x)), float(np.ptp(y))
        if xr > 1e-12 and yr > 1e-12:
            xn = (x - x.min()) / xr
            yn = (y - y.min()) / yr
            signed = (xn - yn) / math.sqrt(2.0)
            interior = np.arange(1, len(frontier) - 1)
            if interior.size:
                j = int(interior[np.argmax(signed[interior])])
                prominence = float(max(0.0, signed[j]))
                diag["pareto_knee_cutoff_m"] = float(frontier[j]["cutoff_m"])
                diag["pareto_knee_prominence"] = prominence
            by_key = {round(float(m["cutoff_m"]), 10): i for i, m in enumerate(frontier)}
            for row in rows:
                idx = by_key.get(round(float(row["cutoff_m"]), 10))
                if idx is not None:
                    row["pareto_x_normalized"] = float(xn[idx])
                    row["pareto_y_normalized"] = float(yn[idx])
                    row["pareto_knee_score"] = float(max(0.0, signed[idx]))
    return rows, diag

def _fallback_changepoint_selection(candidates: list[dict]) -> tuple[dict, str]:
    mode = str(CHANGEPOINT_FALLBACK).strip().lower()
    if mode == "best_marginal_efficiency":
        finite = [m for m in candidates if np.isfinite(m.get("incremental_qualified_share_pct", np.nan))]
        if finite:
            chosen = max(finite, key=lambda m: (float(m["incremental_qualified_share_pct"]),
                                                float(m["qualified_area_m2"]),
                                                -float(m["cutoff_m"])))
            return dict(chosen), "fallback_best_marginal_efficiency"
        mode = "min_disqualified"
    if mode == "min_disqualified":
        chosen = min(candidates, key=lambda m: (float(m["disqualified_pct"]),
                                                 -float(m["qualified_area_m2"]),
                                                 -float(m["cutoff_m"])))
        return dict(chosen), "fallback_min_disqualified"
    if mode == "max_qualified_area":
        chosen = max(candidates, key=lambda m: (float(m["qualified_area_m2"]),
                                                 -float(m["disqualified_pct"]),
                                                 float(m["cutoff_m"])))
        return dict(chosen), "fallback_max_qualified_area"
    raise ValueError("CHANGEPOINT_FALLBACK must be 'best_marginal_efficiency', 'min_disqualified', or 'max_qualified_area'")

def _select_cutoff_marginal_changepoint(metrics: list[dict]) -> tuple[list[dict], dict]:
    """Select last cutoff in the high-efficiency regime using BIC piecewise regression."""
    if str(SELECTION_METHOD).strip().lower() != "marginal_changepoint_bic":
        raise ValueError("SELECTION_METHOD must be 'marginal_changepoint_bic'")

    rows = _add_incremental_metrics(metrics)
    rows, pareto_diag = _annotate_pareto_diagnostic(rows)

    # Optional hard constraint applies only to actual coverage/rejection failures.
    cap_satisfied = True
    eligible = [m for m in rows if int(m["polygon_cells"]) > 0]
    if not eligible:
        raise RuntimeError("All HAND cutoff polygons are empty.")
    if MAX_COVERAGE_FAILURE_PCT is not None:
        constrained = [m for m in eligible if float(m.get("coverage_failure_pct", np.inf)) <= float(MAX_COVERAGE_FAILURE_PCT) + 1e-12]
        if constrained:
            eligible = constrained
        else:
            cap_satisfied = False
    eligible_keys = {round(float(m["cutoff_m"]), 10) for m in eligible}
    for row in rows:
        row["optimization_cap_eligible"] = round(float(row["cutoff_m"]), 10) in eligible_keys
        row["changepoint_series_raw_pct"] = row.get("incremental_qualified_share_pct", np.nan)
        row["changepoint_series_smoothed_pct"] = np.nan
        row["changepoint_single_fit_pct"] = np.nan
        row["changepoint_piecewise_fit_pct"] = np.nan
        row["changepoint_selected_break"] = False
        row["hard_cap_satisfied_by_any_candidate"] = bool(cap_satisfied)

    fit_rows = [m for m in eligible if np.isfinite(m.get("incremental_qualified_share_pct", np.nan))]
    fit_rows.sort(key=lambda m: float(m["cutoff_m"]))
    min_seg = int(CHANGEPOINT_MIN_SEGMENT_POINTS)
    if min_seg < 2:
        raise ValueError("CHANGEPOINT_MIN_SEGMENT_POINTS must be >= 2")

    cp_found = False
    cp_cutoff = None
    cp_index = None
    delta_bic = np.nan
    bic_single = np.nan
    bic_piecewise = np.nan
    left_slope = right_slope = np.nan
    left_mean = right_mean = np.nan
    selection_mode = None
    selection_status = "NO_CLEAR_CHANGEPOINT"

    if len(fit_rows) >= 2 * min_seg:
        x = np.asarray([float(m["cutoff_m"]) for m in fit_rows], dtype=float)
        y_raw = np.asarray([float(m["incremental_qualified_share_pct"]) for m in fit_rows], dtype=float)
        y_fit = _rolling_median_finite(y_raw, int(CHANGEPOINT_MEDIAN_WINDOW))
        sse1, int1, slope1, fitted1 = _linear_fit_sse(x, y_fit)
        bic_single = _bic_from_sse(sse1, len(x), 2)

        candidates = []
        # split index = last point in left/high-efficiency regime; right begins at split+1
        for split in range(min_seg - 1, len(x) - min_seg):
            xl, yl = x[:split + 1], y_fit[:split + 1]
            xr, yr = x[split + 1:], y_fit[split + 1:]
            ssel, il, sl, fitl = _linear_fit_sse(xl, yl)
            sser, ir, sr, fitr = _linear_fit_sse(xr, yr)
            sse2 = ssel + sser
            bic2 = _bic_from_sse(sse2, len(x), 5)  # 4 line coefficients + selected breakpoint
            lmean, rmean = float(np.mean(yl)), float(np.mean(yr))
            # We seek a transition from more-efficient to less-efficient expansion.
            if rmean < lmean - 1e-12:
                candidates.append((bic2, split, sse2, il, sl, ir, sr, fitl, fitr, lmean, rmean))

        if candidates:
            candidates.sort(key=lambda z: z[0])
            bic_piecewise, cp_index, _, il, left_slope, ir, right_slope, fitl, fitr, left_mean, right_mean = candidates[0]
            delta_bic = float(bic_single - bic_piecewise)
            # BIC itself supplies the complexity penalty. A positive delta means the
            # two-segment model is preferred without a custom sensitivity threshold.
            cp_found = bool(delta_bic > 0.0)
            cp_cutoff = float(x[cp_index]) if cp_found else None

            # Write fit diagnostics back to full rows.
            fit_by_key = {round(float(v), 10): i for i, v in enumerate(x)}
            piece = np.full_like(y_fit, np.nan)
            piece[:cp_index + 1] = fitl
            piece[cp_index + 1:] = fitr
            for row in rows:
                idx = fit_by_key.get(round(float(row["cutoff_m"]), 10))
                if idx is not None:
                    row["changepoint_series_smoothed_pct"] = float(y_fit[idx])
                    row["changepoint_single_fit_pct"] = float(fitted1[idx])
                    row["changepoint_piecewise_fit_pct"] = float(piece[idx])

    if cp_found and cp_cutoff is not None:
        selected = next(dict(m) for m in rows if math.isclose(float(m["cutoff_m"]), cp_cutoff, abs_tol=1e-9))
        selection_mode = "marginal_changepoint_bic"
        selection_status = "INTERIOR_CHANGEPOINT"
    else:
        selected, selection_mode = _fallback_changepoint_selection(eligible)
        selection_status = "NO_CLEAR_CHANGEPOINT"

    selected_key = round(float(selected["cutoff_m"]), 10)
    min_cutoff = min(float(m["cutoff_m"]) for m in eligible)
    max_cutoff = max(float(m["cutoff_m"]) for m in eligible)
    endpoint_flag = None
    if math.isclose(float(selected["cutoff_m"]), min_cutoff, abs_tol=1e-9):
        endpoint_flag = "MINIMUM_BOUNDARY_SELECTED"
    elif math.isclose(float(selected["cutoff_m"]), max_cutoff, abs_tol=1e-9):
        endpoint_flag = "MAXIMUM_BOUNDARY_SELECTED"
    if endpoint_flag is not None:
        selection_status = f"{selection_status}__{endpoint_flag}"

    if cp_found and cp_cutoff is not None:
        for row in rows:
            if math.isclose(float(row["cutoff_m"]), cp_cutoff, abs_tol=1e-9):
                row["changepoint_selected_break"] = True

    min_disq = min(rows, key=lambda m: (float(m["disqualified_pct"]), -float(m["qualified_area_m2"]), -float(m["cutoff_m"])))
    max_good = max(rows, key=lambda m: (float(m["qualified_area_m2"]), -float(m["disqualified_pct"]), float(m["cutoff_m"])))

    selected_ann = next(dict(m) for m in rows if round(float(m["cutoff_m"]), 10) == selected_key)
    selected_ann.update({
        "selection_method": str(SELECTION_METHOD),
        "selection_mode_used": selection_mode,
        "selection_status": selection_status,
        "selection_reason": (
            "BIC preferred a two-segment decline in incremental qualified-area efficiency; selected the last cutoff in the higher-efficiency regime"
            if cp_found else
            f"BIC did not support a declining two-segment efficiency model; {selection_mode}"
        ),
        "changepoint_cutoff_m": cp_cutoff,
        "changepoint_bic_single_line": float(bic_single) if np.isfinite(bic_single) else None,
        "changepoint_bic_piecewise": float(bic_piecewise) if np.isfinite(bic_piecewise) else None,
        "changepoint_delta_bic_single_minus_piecewise": float(delta_bic) if np.isfinite(delta_bic) else None,
        "changepoint_bic_prefers_piecewise": bool(cp_found),
        "changepoint_left_mean_incremental_qualified_pct": float(left_mean) if np.isfinite(left_mean) else None,
        "changepoint_right_mean_incremental_qualified_pct": float(right_mean) if np.isfinite(right_mean) else None,
        "changepoint_left_slope_pct_per_m": float(left_slope) if np.isfinite(left_slope) else None,
        "changepoint_right_slope_pct_per_m": float(right_slope) if np.isfinite(right_slope) else None,
        "changepoint_min_segment_points": int(min_seg),
        "changepoint_median_window": int(CHANGEPOINT_MEDIAN_WINDOW),
        "diagnostic_min_disqualified_cutoff_m": float(min_disq["cutoff_m"]),
        "diagnostic_min_disqualified_pct": float(min_disq["disqualified_pct"]),
        "diagnostic_max_qualified_area_cutoff_m": float(max_good["cutoff_m"]),
        "diagnostic_max_qualified_area_m2": float(max_good["qualified_area_m2"]),
        "hard_max_coverage_failure_pct": MAX_COVERAGE_FAILURE_PCT,
        "hard_cap_satisfied_by_any_candidate": bool(cap_satisfied),
        **pareto_diag,
    })
    selected_ann["selected"] = True
    for row in rows:
        row["selected"] = round(float(row["cutoff_m"]), 10) == selected_key
    return rows, selected_ann

# ============================================================================
# NEW SIMPLE SINGLE-NETWORK HAND INPUT PIPELINE
# ============================================================================

def validate_dem_and_network():
    for p in (STREAM_NETWORK, POST_DEM, DOD):
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")
    with rasterio.open(POST_DEM) as ds:
        if ds.count < 1 or ds.crs is None:
            raise ValueError("Post DEM requires one elevation band and a valid CRS")
        if ds.crs.is_geographic:
            raise ValueError("Post DEM CRS is geographic. Reproject DEM to a metric projected CRS first.")
        # EPSG common use metric; use pyproj for robust unit validation
        from pyproj import CRS
        units = [a.unit_name.lower() for a in CRS.from_user_input(ds.crs).axis_info[:2]]
        if units and any(u not in ('metre','meter','metres','meters') for u in units):
            raise ValueError(f"Post DEM horizontal CRS must use metres; CRS units = {units}")
        if ds.transform.b != 0 or ds.transform.d != 0:
            raise ValueError("Rotated DEM grids are not supported")
        if not (ds.transform.a > 0 and ds.transform.e < 0):
            raise ValueError("Expected north-up DEM raster")
        a = ds.read(1, masked=True).filled(np.nan).astype(np.float32)
        valid = np.isfinite(a)
        if not valid.any():
            raise ValueError("No valid post-DEM elevations")
        profile = ds.profile.copy()
    gdf = gpd.read_file(STREAM_NETWORK, layer=STREAM_LAYER) if STREAM_LAYER else gpd.read_file(STREAM_NETWORK)
    if gdf.empty or gdf.crs is None:
        raise ValueError("Stream network is empty or has no defined CRS")
    if not all(g.geom_type in ('LineString', 'MultiLineString') for g in gdf.geometry.dropna()):
        raise ValueError("The stream input must contain LineString/MultiLineString features")
    gdf = gdf.to_crs(profile['crs'])
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    if STREAM_BUFFER_M > 0:
        geom_list = [g.buffer(STREAM_BUFFER_M) for g in gdf.geometry]
    else:
        geom_list = list(gdf.geometry)
    net = rasterize(((g, 1) for g in geom_list), out_shape=a.shape,
                    transform=profile['transform'], fill=0, dtype='uint8',
                    all_touched=True).astype(bool) & valid
    if not net.any():
        raise ValueError("The stream network does not intersect valid DEM cells")
    return a, valid, net, profile, gdf


def d8_hand(dem, valid, stream, transform):
    """Route strictly downhill to supplied stream cells using steepest D8 descent.

    Strict descent prevents circular pointers. Sinks, flats, and DEM cells whose
    downhill flow does not reach the provided network are excluded from HAND.
    This is intentionally an unconditioned, simple adaptation.
    """
    h, w = dem.shape
    z = dem.astype(np.float64)
    next_index = np.full((h,w), -1, dtype=np.int64)
    best_slope = np.zeros((h,w), dtype=np.float64)
    sx, sy = abs(transform.a), abs(transform.e)
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr == 0 and dc == 0: continue
            r0, r1 = max(0,-dr), min(h,h-dr)
            c0, c1 = max(0,-dc), min(w,w-dc)
            rr0, rr1 = r0+dr, r1+dr
            cc0, cc1 = c0+dc, c1+dc
            if r1 <= r0 or c1 <= c0: continue
            a = z[r0:r1,c0:c1]
            b = z[rr0:rr1,cc0:cc1]
            feasible = valid[r0:r1,c0:c1] & valid[rr0:rr1,cc0:cc1] & ~stream[r0:r1,c0:c1]
            slope = (a-b)/math.hypot(dr*sy,dc*sx)
            update = feasible & (slope > best_slope[r0:r1,c0:c1])
            if update.any():
                region = best_slope[r0:r1,c0:c1]
                region[update] = slope[update]
                region = next_index[r0:r1,c0:c1]
                rgrid,cgrid = np.indices(a.shape)
                region[update] = (rgrid[update]+rr0)*w+(cgrid[update]+cc0)
    # Sort ascending: if A flows to B, B is lower and was evaluated first.
    ids = np.flatnonzero(valid.ravel())
    ids = ids[np.argsort(z.ravel()[ids],kind='stable')]
    anchor = np.full(z.size, -1, dtype=np.int64)
    anchors = np.flatnonzero(stream.ravel())
    anchor[anchors] = anchors
    flat_next = next_index.ravel()
    flat_stream = stream.ravel()
    for i in ids:
        if flat_stream[i]: continue
        recipient = flat_next[i]
        if recipient >= 0:
            anchor[i] = anchor[recipient]
    ok = valid.ravel() & (anchor >= 0)
    out = np.full(z.size,np.nan,dtype=np.float32)
    finite_ids = np.flatnonzero(ok)
    out[finite_ids] = np.maximum(0, z.ravel()[finite_ids] - z.ravel()[anchor[finite_ids]])
    return out.reshape(z.shape), ok.reshape(z.shape)


def nearest_stream_hand(dem, valid, stream, transform):
    _, indices = ndimage.distance_transform_edt(~stream, sampling=(abs(transform.e), abs(transform.a)),
                                                 return_indices=True)
    stream_z = dem[tuple(indices)]
    hand = np.maximum(0.0,dem - stream_z).astype(np.float32)
    hand[~valid] = np.nan
    return hand, valid.copy()


def dod_on_dem_grid(profile):
    with rasterio.open(DOD) as ds:
        if ds.crs is None:
            raise ValueError("DoD requires a defined CRS")
        if ds.crs.is_geographic and profile['crs'].is_geographic:
            raise ValueError("Raster CRS must be projected")
        source = ds.read(1, masked=True).filled(np.nan).astype(np.float32)
        aligned = (ds.crs==profile['crs'] and ds.shape==(profile['height'],profile['width'])
                   and ds.transform==profile['transform'])
        if aligned:
            return source, False
        dest = np.full((profile['height'], profile['width']),np.nan,dtype=np.float32)
        reproject(source, dest, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=profile['transform'],dst_crs=profile['crs'],
                  src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.nearest)
        return dest, True


def classify_dod(dod):
    t = float(MIN_ABS_CHANGE_M)
    if not np.isfinite(t) or t < 0:
        raise ValueError('MIN_ABS_CHANGE_M must be nonnegative')
    mode = CHANGE_MODE.lower().strip()
    if mode not in ('either','deposition','erosion'):
        raise ValueError('CHANGE_MODE must be either, deposition, or erosion')
    base = np.isfinite(dod)
    extreme = np.zeros_like(base)
    if MAX_ABS_DOD_CHANGE_M is not None:
        if MAX_ABS_DOD_CHANGE_M <= 0: raise ValueError('MAX_ABS_DOD_CHANGE_M must be positive')
        extreme = base & (np.abs(dod)>float(MAX_ABS_DOD_CHANGE_M))
    eligible = base & ~extreme
    if mode=='either': qualified = eligible & (np.abs(dod)>t)
    elif mode=='deposition': qualified = eligible & (dod>t)
    else: qualified = eligible & (dod < -t)
    below = eligible & ~qualified
    missing = ~base
    # Use original optimizer names for output schema compatibility.
    # below_lod is actually below user-set magnitude criterion, NOT LoD95.
    return {'qualified':qualified,'below_lod':below,'missing':missing,
            'extreme':extreme,'disqualified':~qualified}


def write_tif(path, data, profile, nodata=-9999.0):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=profile.copy()
    p.update(driver='GTiff',count=1,dtype='float32',nodata=nodata,compress='deflate',BIGTIFF='IF_SAFER')
    for field in ('blockxsize','blockysize','tiled','photometric','interleave','predictor'):
        p.pop(field,None)
    with rasterio.open(path,'w',**p) as dst:
        dst.write(np.where(np.isfinite(data), data, nodata).astype(np.float32),1)


def write_mask(path, data, profile):
    p=profile.copy()
    p.update(driver='GTiff',count=1,dtype='uint8',nodata=0,compress='deflate',BIGTIFF='IF_SAFER')
    for field in ('blockxsize','blockysize','tiled','photometric','interleave','predictor'):
        p.pop(field,None)
    with rasterio.open(path,'w',**p) as dst:
        dst.write(data.astype(np.uint8),1)


def dump_geoms(path, geometries, crs):
    rows=[{'product':key,'geometry':g} for key,g in geometries.items() if g is not None and not g.is_empty]
    if not rows: raise RuntimeError('No selected geometry could be exported')
    gpd.GeoDataFrame(rows,geometry='geometry',crs=crs).to_file(path,driver='GPKG',layer='channel_envelopes')


def safe_json(o):
    if isinstance(o,dict): return {k:safe_json(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [safe_json(v) for v in o]
    if isinstance(o,Path): return str(o)
    if isinstance(o,np.generic): return safe_json(o.item())
    if isinstance(o,float) and not np.isfinite(o): return None
    return o


def make_figure(out, dem, dod, stream, hand, valid, rows, selected, raw, env, profile):
    fig, ax=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    extent=(profile['transform'].c,profile['transform'].c+profile['width']*profile['transform'].a,
            profile['transform'].f+profile['height']*profile['transform'].e,profile['transform'].f)
    a=ax[0,0]; a.imshow(dem,extent=extent,cmap='terrain');
    a.contour(stream,levels=[.5],colors='cyan',linewidths=.7,extent=extent)
    a.set_title('Post-event DEM and supplied stream cells')
    a=ax[0,1]; im=a.imshow(np.ma.masked_invalid(hand),extent=extent,cmap='viridis',vmin=0,
                          vmax=max(HAND_CUTOFF_MAX_M,1))
    fig.colorbar(im,ax=a,label='HAND (m)',shrink=.75)
    a.set_title('Computed HAND (unconnected excluded)')
    a=ax[1,0]; a.imshow(np.ma.masked_invalid(dod),extent=extent,cmap='RdBu',vmin=-2,vmax=2)
    a.contour(env,levels=[.5],colors='black',linewidths=.8,extent=extent)
    a.set_title('DoD with selected envelope outline')
    a=ax[1,1]
    x=[r['cutoff_m'] for r in rows]
    y=[r['incremental_qualified_share_pct'] for r in rows]
    a.plot(x,y,'o-',ms=2,label='Marginal qualifying share')
    a.plot(x,[r['changepoint_piecewise_fit_pct'] for r in rows],label='BIC piecewise fit')
    a.axvline(selected['cutoff_m'],color='black',linestyle='--',label='Selected cutoff')
    a.set(xlabel='HAND cutoff (m)',ylabel='Marginal qualifying share (%)',title=selected['selection_status'])
    a.legend(fontsize=8);a.grid(alpha=.2)
    fig.savefig(out,dpi=160,bbox_inches='tight')
    plt.close(fig)


def main():
    print('[Simple channel optimizer] Checking inputs...')
    dem, dem_valid, stream, profile, network = validate_dem_and_network()
    out=OUTPUT_DIR
    out.mkdir(parents=True,exist_ok=True)
    if not OVERWRITE_OUTPUTS and (out/'summary.json').exists():
        raise FileExistsError(f'Existing output directory: {out}')
    print(f'[HAND] {HAND_METHOD}; grid {dem.shape}; stream cells {int(stream.sum()):,}')
    if HAND_METHOD.upper()=='ROUTED_D8': hand,hand_valid=d8_hand(dem,dem_valid,stream,profile['transform'])
    elif HAND_METHOD.upper()=='NEAREST_STREAM':hand,hand_valid=nearest_stream_hand(dem,dem_valid,stream,profile['transform'])
    else: raise ValueError('HAND_METHOD must be ROUTED_D8 or NEAREST_STREAM')
    print(f'[HAND] valid connected cells: {int(hand_valid.sum()):,} / {int(dem_valid.sum()):,}')
    if not hand_valid.any(): raise RuntimeError('No DEM cells connected to stream')
    dod,reprojected=dod_on_dem_grid(profile)
    quality=classify_dod(dod)
    write_tif(out/'HAND_m.tif',hand,profile)
    write_mask(out/'stream_cells.tif',stream,profile)
    write_mask(out/'HAND_valid_mask.tif',hand_valid,profile)
    qcode=np.zeros(dem.shape,dtype=np.uint8)
    qcode[quality['missing']]=4
    qcode[quality['extreme']]=3
    qcode[quality['below_lod']]=2
    qcode[quality['qualified']]=1
    write_mask(out/'DoD_quality_classes.tif',qcode,profile)
    measurements=[]
    print(f'[Optimize] {len(cutoff_values())} candidate cutoffs')
    for cutoff in cutoff_values():
        candidate=_candidate_for_cutoff(hand,hand_valid,profile,cutoff)
        metrics=_quality_metrics(candidate,quality)
        measurements.append(metrics)
    results, selected=_select_cutoff_marginal_changepoint(measurements)
    key=selected['cutoff_m']
    chosen=_candidate_for_cutoff(hand,hand_valid,profile,key)
    write_mask(out/'selected_raw_HAND_mask.tif',chosen['raw_mask'],profile)
    write_mask(out/'selected_envelope_mask.tif',chosen['eval_mask'],profile)
    dump_geoms(out/'selected_channel_polygons.gpkg',
               {'raw_HAND_threshold':chosen['raw_geom'], 'regularized_envelope':chosen['envelope_geom']},
               profile['crs'])
    # Shapefile included for simple GIS sharing.
    env=chosen['envelope_geom']
    if env is not None and not env.is_empty:
        gpd.GeoDataFrame([{'cutoff_m':float(key),'geometry':env}],geometry='geometry',crs=profile['crs']).to_file(
            out/'selected_channel_envelope.shp')
    if WRITE_CANDIDATE_GPKG:
        rows=[]
        for m in results:
            candidate=_candidate_for_cutoff(hand,hand_valid,profile,m['cutoff_m'])
            geom=candidate['envelope_geom']
            if geom is not None and not geom.is_empty:
                rows.append({'cutoff_m':m['cutoff_m'],'qualified_pct':m['qualified_change_pct'],'geometry':geom})
        if rows:
            gpd.GeoDataFrame(rows,geometry='geometry',crs=profile['crs']).to_file(
                out/'all_candidate_envelopes.gpkg',driver='GPKG')
    df=pd.DataFrame(results).rename(columns={
        'below_lod_cells':'below_magnitude_threshold_cells',
        'below_lod_area_m2':'below_magnitude_threshold_area_m2',
        'below_lod_pct':'below_magnitude_threshold_pct',
        'incremental_below_lod_share_pct':'incremental_below_magnitude_share_pct',
        'delta_below_lod_area_m2':'delta_below_magnitude_area_m2',
    })
    df.to_csv(out/'cutoff_metrics.csv',index=False)
    if WRITE_QC_FIGURE:
        make_figure(out/'optimization_QC.png',dem,dod,stream,hand,hand_valid,
                    results,selected,chosen['raw_mask'],chosen['eval_mask'],profile)
    summary={
        'source_stream_network':str(STREAM_NETWORK),'post_dem':str(POST_DEM),'dod':str(DOD),
        'stream_layer':STREAM_LAYER,'hand_method':HAND_METHOD,
        'dod_resampled_to_dem':reprojected,
        'classification':'DoD magnitude proxy; no LoD95 uncertainty threshold',
        'change_mode':CHANGE_MODE,'min_abs_change_m':MIN_ABS_CHANGE_M,
        'max_abs_dod_change_m':MAX_ABS_DOD_CHANGE_M,
        'connected_hand_cells':int(hand_valid.sum()),'dem_valid_cells':int(dem_valid.sum()),
        'stream_cells':int(stream.sum()),'selected_cutoff_m':float(key),
        'selected':selected,
        'note':'Unconditioned DEM D8 differs from original CO2CHAIN routing. Check alignment, sinks, and DoD sign convention.'
    }
    (out/'summary.json').write_text(json.dumps(safe_json(summary),indent=2),encoding='utf-8')
    print('[SUCCESS] Selected cutoff:',key,'m; status:',selected['selection_status'])
    print('[OUTPUT]',out)

if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        raise

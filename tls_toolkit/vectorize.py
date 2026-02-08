# tls_toolkit/vectorize.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import csv
import json

from .constants import ID2CLASS, ID2COLOR_RGB
from .utils import ensure_dir


@dataclass
class VectorizeResult:
    geojson_path: str
    qupath_geojson_path: str
    objects_table_path: str
    stats: Dict[str, Any]


def _contour_to_ring(cnt: np.ndarray, coord_scale: float) -> List[List[float]]:
    """OpenCV contour (Nx1x2) -> GeoJSON ring, closed."""
    s = float(coord_scale)
    ring: List[List[float]] = []
    for pt in cnt:
        x, y = pt[0]
        ring.append([float(x) * s, float(y) * s])
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _find_external_polygons(binary_u8: np.ndarray, min_object_area: int) -> List[np.ndarray]:
    """
    Return list of external contours (CHAIN_APPROX_NONE), filtered by contour area.
    Using RETR_EXTERNAL only to keep topology simple and QuPath-stable.
    """
    if int(binary_u8.sum()) == 0:
        return []
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for cnt in contours or []:
        area = float(cv2.contourArea(cnt))
        if area >= float(min_object_area):
            out.append(cnt)
    return out


def _build_class_masks_for_vectorization(
    full_mask: np.ndarray,
    tls_label: int,
    gc_label: int,
    immune_label: int,
    *,
    tls_includes_gc: bool = True,
    export_gc: bool = True,
) -> List[Tuple[int, str, np.ndarray]]:
    """
    Build per-class binary masks used for polygonization.

    Crucial semantics:
      - TLS geometry should include GC and expand already happened in postprocess.
        Here we implement "TLS polygon contains GC" as:
            TLS_geom = TLS_mask OR GC_mask
        (only if tls_includes_gc=True)

      - ImmuneCluster stays independent (mask == immune_label)

      - GC can optionally be exported as its own class.
    """
    tls = (full_mask == int(tls_label)).astype(np.uint8)
    gc = (full_mask == int(gc_label)).astype(np.uint8)
    immune = (full_mask == int(immune_label)).astype(np.uint8)

    items: List[Tuple[int, str, np.ndarray]] = []

    # TLS: union with GC if requested
    tls_geom = np.maximum(tls, gc) if tls_includes_gc else tls
    items.append((int(tls_label), ID2CLASS[int(tls_label)], tls_geom))

    # GC
    if export_gc:
        items.append((int(gc_label), ID2CLASS[int(gc_label)], gc))

    # ImmuneCluster
    items.append((int(immune_label), ID2CLASS[int(immune_label)], immune))
    return items


def vectorize_mask_to_geojson(
    full_mask: np.ndarray,
    out_dir: str,
    min_object_area: int = 40000,
    *,
    coord_scale: float = 1.0,
    tls_label: int = 1,
    gc_label: int = 2,
    immune_label: int = 3,
    tls_includes_gc: bool = True,
    export_gc: bool = True,
) -> VectorizeResult:
    """
    Output TWO GeoJSON files with IDENTICAL geometry (same polygons, same IDs):
      1) annotations.geojson        : includes measurements/statistics
      2) annotations_qupath.geojson : strict QuPath schema, minimal properties

    Geometry source:
      - external contours only (RETR_EXTERNAL)
      - no simplification (to preserve exact mask boundary)
      - topology is stabilized already at mask-level by postprocess.py
    """
    out_dir = ensure_dir(out_dir)
    vec_dir = ensure_dir(Path(out_dir) / "artifacts" / "vector")
    geojson_path = str(Path(vec_dir) / "annotations.geojson")
    qupath_geojson_path = str(Path(vec_dir) / "annotations_qupath.geojson")
    table_path = str(Path(vec_dir) / "objects_table.csv")

    s = float(coord_scale)

    # Build class masks (TLS can include GC)
    class_items = _build_class_masks_for_vectorization(
        full_mask=full_mask,
        tls_label=tls_label,
        gc_label=gc_label,
        immune_label=immune_label,
        tls_includes_gc=tls_includes_gc,
        export_gc=export_gc,
    )

    # We will create ONE canonical list of objects (geometry+id),
    # then write two GeoJSONs with different properties.
    objects: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    fid = 0
    dropped_bad = 0

    for class_id, class_name, binmask in class_items:
        if class_id not in ID2COLOR_RGB:
            continue

        contours = _find_external_polygons(binmask, min_object_area=min_object_area)
        if not contours:
            continue

        for cnt in contours:
            ring = _contour_to_ring(cnt, coord_scale=s)
            if len(ring) < 4:
                dropped_bad += 1
                continue

            fid += 1
            feature_id = f"{class_name}_{fid:06d}"

            area = float(cv2.contourArea(cnt)) * (s ** 2)
            peri = float(cv2.arcLength(cnt, True)) * s
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = float(M["m10"] / M["m00"]) if M["m00"] != 0 else float(x + w / 2.0)
            cy = float(M["m01"] / M["m00"]) if M["m00"] != 0 else float(y + h / 2.0)

            geom = {"type": "Polygon", "coordinates": [ring]}
            objects.append({
                "object_id": feature_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "geometry": geom,
                "area_px": area,
                "perimeter_px": peri,
                "bbox_x": float(x) * s,
                "bbox_y": float(y) * s,
                "bbox_w": float(w) * s,
                "bbox_h": float(h) * s,
                "centroid_x": float(cx) * s,
                "centroid_y": float(cy) * s,
            })

            rows.append({
                "object_id": feature_id,
                "class_id": int(class_id),
                "class_name": class_name,
                "area_px": float(area),
                "perimeter_px": float(peri),
                "bbox_x": float(x) * s,
                "bbox_y": float(y) * s,
                "bbox_w": float(w) * s,
                "bbox_h": float(h) * s,
                "centroid_x": float(cx) * s,
                "centroid_y": float(cy) * s,
                "holes": 0,
            })

    # ---- annotations.geojson (rich) ----
    rich_features: List[Dict[str, Any]] = []
    for obj in objects:
        cid = int(obj["class_id"])
        cname = str(obj["class_name"])
        r, g, b = ID2COLOR_RGB[cid]
        rich_features.append({
            "type": "Feature",
            "id": obj["object_id"],
            "properties": {
                "objectType": "annotation",
                "classification": {"name": cname, "color": [int(r), int(g), int(b)]},
                "isLocked": False,
                "measurements": [
                    {"name": "area_px", "value": float(obj["area_px"])},
                    {"name": "perimeter_px", "value": float(obj["perimeter_px"])},
                    {"name": "bbox_x", "value": float(obj["bbox_x"])},
                    {"name": "bbox_y", "value": float(obj["bbox_y"])},
                    {"name": "bbox_w", "value": float(obj["bbox_w"])},
                    {"name": "bbox_h", "value": float(obj["bbox_h"])},
                    {"name": "centroid_x", "value": float(obj["centroid_x"])},
                    {"name": "centroid_y", "value": float(obj["centroid_y"])},
                ],
            },
            "geometry": obj["geometry"],
        })

    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": rich_features}, f, ensure_ascii=False, indent=2)

    # ---- annotations_qupath.geojson (strict/minimal) ----
    qp_features: List[Dict[str, Any]] = []
    for obj in objects:
        cid = int(obj["class_id"])
        cname = str(obj["class_name"])
        r, g, b = ID2COLOR_RGB[cid]
        qp_features.append({
            "type": "Feature",
            "id": obj["object_id"],
            "properties": {
                "objectType": "annotation",
                "classification": {"name": cname, "color": [int(r), int(g), int(b)]},
                "isLocked": False,
                "measurements": [],
            },
            "geometry": obj["geometry"],
        })

    with open(qupath_geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": qp_features}, f, ensure_ascii=False, indent=2)

    # ---- objects_table.csv ----
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "object_id", "class_id", "class_name", "area_px", "perimeter_px",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h", "centroid_x", "centroid_y", "holes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    stats = {
        "objects_total": int(len(rows)),
        "features_total": int(len(rich_features)),
        "qupath_objects_total": int(len(qp_features)),
        "min_object_area": int(min_object_area),
        "coord_scale_to_level0": float(s),
        "dropped_bad": int(dropped_bad),

        "tls_includes_gc": bool(tls_includes_gc),
        "export_gc": bool(export_gc),
        "geometry_policy": "EXTERNAL_ONLY; NO_SIMPLIFY; geometry identical between geojson and qupath_geojson",
    }

    return VectorizeResult(
        geojson_path=geojson_path,
        qupath_geojson_path=qupath_geojson_path,
        objects_table_path=table_path,
        stats=stats,
    )

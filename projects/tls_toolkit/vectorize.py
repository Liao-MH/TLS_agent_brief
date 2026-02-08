from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

import numpy as np
import cv2
import csv
import json

from .constants import ID2CLASS, ID2COLOR_RGB, DEFAULT_CONTOUR_SIMPLIFY_EPS
from .utils import ensure_dir


@dataclass
class VectorizeResult:
    geojson_path: str
    qupath_geojson_path: str
    objects_table_path: str
    stats: Dict[str, Any]


def _simplify_contour(cnt: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        return cnt
    e = float(eps)
    approx = cv2.approxPolyDP(cnt, epsilon=e, closed=True)
    if approx is None or len(approx) < 4:
        return cnt
    return approx


def _contour_to_coords(cnt: np.ndarray, coord_scale: float = 1.0) -> List[List[float]]:
    coords: List[List[float]] = []
    s = float(coord_scale) if coord_scale is not None else 1.0
    for pt in cnt:
        x, y = pt[0]
        coords.append([float(x) * s, float(y) * s])
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords

def _write_qupath_geojson(
    full_mask: np.ndarray,
    out_path: str,
    min_object_area: int,
    coord_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Write a QuPath-import-friendly GeoJSON (strict schema).
    Template matches user's proven working wsi_inference.py output:
      - RETR_EXTERNAL only (no holes)
      - Polygon coordinates: [coords] (single ring)
      - properties: objectType/classification/isLocked/measurements=[]
    """
    features: List[Dict[str, Any]] = []
    fid = 0

    for class_id, class_name in ID2CLASS.items():
        if class_id not in ID2COLOR_RGB:
            continue

        cls_mask = (full_mask == int(class_id)).astype(np.uint8)
        if int(cls_mask.sum()) < int(min_object_area):
            continue

        contours, _ = cv2.findContours(cls_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < float(min_object_area):
                continue

            coords = _contour_to_coords(cnt, coord_scale=coord_scale)
            if not coords:
                continue

            fid += 1
            r, g, b = ID2COLOR_RGB[int(class_id)]
            feat = {
                "type": "Feature",
                "id": f"{class_name}_{fid:06d}",
                "properties": {
                    "objectType": "annotation",
                    "classification": {"name": class_name, "color": [int(r), int(g), int(b)]},
                    "isLocked": False,
                    "measurements": [],
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
            features.append(feat)

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    return {"qupath_objects_total": int(len(features))}




def vectorize_mask_to_geojson(
    full_mask: np.ndarray,
    out_dir: str,
    min_object_area: int,
    simplify_eps: float = DEFAULT_CONTOUR_SIMPLIFY_EPS,
    coord_scale: float = 1.0,
) -> VectorizeResult:
    """
    Vectorize each class mask into GeoJSON with holes support (via RETR_CCOMP),
    and output an objects_table.csv for downstream analysis.

    Geometry:
      - Emit Polygon for each external contour; holes are included as inner rings if present.
    """
    out_dir = ensure_dir(out_dir)
    vec_dir = ensure_dir(Path(out_dir) / "artifacts" / "vector")
    geojson_path = str(Path(vec_dir) / "annotations.geojson")
    qupath_geojson_path = str(Path(vec_dir) / "annotations_qupath.geojson")
    table_path = str(Path(vec_dir) / "objects_table.csv")

    features: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    fid = 0
    s = float(coord_scale) if coord_scale is not None else 1.0

    for class_id, class_name in ID2CLASS.items():
        if class_id not in ID2COLOR_RGB:
            continue

        cls_mask = (full_mask == int(class_id)).astype(np.uint8)
        if int(cls_mask.sum()) < int(min_object_area):
            continue

        contours, hierarchy = cv2.findContours(cls_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None or len(contours) == 0:
            continue

        hierarchy = hierarchy[0]

        for i, cnt in enumerate(tqdm(
            contours,
            total=len(contours),
            desc=f"Vectorize {class_name}",
            dynamic_ncols=True,
        )):
            parent = int(hierarchy[i][3])
            if parent != -1:
                continue

            area = float(cv2.contourArea(cnt))
            if area < float(min_object_area):
                continue

            fid += 1

            ext = _simplify_contour(cnt, simplify_eps)
            ext_coords = _contour_to_coords(ext, coord_scale=s)

            holes_coords: List[List[List[float]]] = []
            child = int(hierarchy[i][2])
            while child != -1:
                hole_cnt = contours[child]
                hole_area = float(cv2.contourArea(hole_cnt))
                if hole_area >= 10.0:
                    hole_s = _simplify_contour(hole_cnt, simplify_eps)
                    holes_coords.append(_contour_to_coords(hole_s, coord_scale=s))
                child = int(hierarchy[child][0])

            rings = [ext_coords] + holes_coords

            x, y, w, h = cv2.boundingRect(cnt)
            peri = float(cv2.arcLength(cnt, True))
            M = cv2.moments(cnt)
            cx = float(M["m10"] / M["m00"]) if M["m00"] != 0 else float(x + w / 2)
            cy = float(M["m01"] / M["m00"]) if M["m00"] != 0 else float(y + h / 2)

            feature_id = f"{class_name}_{fid:06d}"
            feat = {
                "type": "Feature",
                "id": feature_id,
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": class_name,
                        "color": [int(c) for c in ID2COLOR_RGB[int(class_id)]],
                    },
                    "isLocked": False,
                    "measurements": [
                        # Measurements are reported in level0 pixel units after scaling.
                        {"name": "area_px", "value": float(area) * (s ** 2)},
                        {"name": "perimeter_px", "value": float(peri) * s},
                    ],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": rings,
                },
            }
            features.append(feat)

            rows.append({
                "object_id": feature_id,
                "class_id": int(class_id),
                "class_name": class_name,
                # Store table fields in level0 pixel units after scaling.
                "area_px": float(area) * (s ** 2),
                "perimeter_px": float(peri) * s,
                "bbox_x": float(x) * s,
                "bbox_y": float(y) * s,
                "bbox_w": float(w) * s,
                "bbox_h": float(h) * s,
                "centroid_x": float(cx) * s,
                "centroid_y": float(cy) * s,
                "holes": int(len(holes_coords)),
            })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
        
    # QuPath-friendly GeoJSON (strict schema)
    qp_stats = _write_qupath_geojson(
        full_mask=full_mask,
        out_path=qupath_geojson_path,
        min_object_area=min_object_area,
        coord_scale=s,
    )


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
        "features_total": int(len(features)),
        "min_object_area": int(min_object_area),
        "simplify_eps": float(simplify_eps),
        "qupath_objects_total": int(qp_stats.get("qupath_objects_total", 0)),
        "coord_scale_to_level0": float(s),
    }
    return VectorizeResult(
        geojson_path=geojson_path,
        qupath_geojson_path=qupath_geojson_path,
        objects_table_path=table_path,
        stats=stats,
    )

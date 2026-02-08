from __future__ import annotations

SCHEMA_VERSION = "1.0.0"

# Class mapping (matches the model convention)
# 0=background, 1=TLS, 2=GC, 3=ImmuneCluster
ID2CLASS = {
    1: "TLS",
    2: "GC",
    3: "ImmuneCluster",
}

ID2COLOR_RGB = {
    1: (0, 255, 0),       # TLS
    2: (235, 143, 124),   # GC
    3: (0, 0, 255),       # ImmuneCluster
}

# Default postprocess rule version
POSTPROC_RULE_VERSION = "tls_gc_rule_v2_proximity"

# Default vectorization settings
DEFAULT_CONTOUR_SIMPLIFY_EPS = 2.0  # pixels; adjust in presets if needed

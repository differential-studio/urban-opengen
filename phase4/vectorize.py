"""
vectorize.py

Urban OpenGen tiles to clean vector geometry: building polygons with heights,
street centrelines with widths, street plates, blocks and green areas.

Input is the 4-channel tile the models read and write (footprint, height,
street, green; signed-distance encoding, byte 128 on the class edge). Output is
plain lists of polygons and polylines in native pixel coordinates (0..W, y down,
grid corners), the same convention as regular.js, plus a GeoJSON export in metres.

Why this is not another tracer
------------------------------
Tracing the outline of a blob and simplifying it reproduces every wobble in the
raster. Instead each building is *fitted*:

  1. its own axis is measured from the gradient of the distance field along its
     edge (angle-doubling vote, so the four sides of a rectangle agree), falling
     back to the nearest street's direction, then the tile's;
  2. in a frame rotated to that axis the distance field is resampled at sub-pixel
     resolution and every boundary sample votes for the line it lies on: pixels
     whose edge runs across the frame vote for a vertical line at their sub-pixel
     zero crossing, pixels whose edge runs along it vote for a horizontal one;
  3. the few lines that collect enough votes define a grid of cells; each cell is
     inside or outside by majority of the field, and the union of the inside cells
     is the building. Every edge lies exactly on a detected line, so a slightly
     crooked 30 m wall becomes one straight 30 m wall, not a staircase;
  4. convex corners that the field says are cut are chamfered (Barcelona);
  5. a building whose height map has two plateaus is split into two parts that
     share the same lines, so a tower on a podium comes out as two boxes that meet.

Streets are the skeleton of the street field turned into a graph, cleaned
(spurs, junction clusters, degree-2 nodes), simplified into straight runs, snapped
to the tile's dominant direction where they are close to it, and re-intersected at
the junctions by least squares. The width of every edge is twice the mean distance
field along it. Plates are the runs buffered by half their width; blocks are the
tile minus the plates. Greens are traced from the field and clipped to the blocks.

Everything works on generated and real tiles alike; `roundtrip()` rasterises the
result back and reports how faithful it is.

Dependencies: numpy, scipy, shapely, pillow (all already required by the project).
"""
from __future__ import annotations

import heapq
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, MultiPolygon, LineString, box
from shapely.ops import unary_union
import shapely

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "phase0"):
    if (cand / "tile_codec.py").exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
from tile_codec import grey_to_metres, metres_to_grey, skeletonize, HeightLUT, FLOOR_HEIGHT_M  # noqa: E402

SDF_TRUNC_PX = 8.0

DEFAULTS = dict(
    # fields
    smooth_px=0.0,          # gaussian blur of the fields before anything else (native px); 0 = off
    # buildings
    min_building_m2=30.0,   # smaller footprints are dropped
    min_hole_m2=40.0,       # smaller courtyards are filled
    up=4,                   # frame resolution, samples per native pixel
    band_px=0.9,            # boundary samples within this distance of the edge vote for lines
    line_sep_m=3.0,         # two lines closer than this are one line
    line_min_m=4.0,         # a line needs at least this much edge behind it
    line_ang_deg=28.0,      # a boundary sample votes only when its edge is within this of an axis
    max_gap_m=18.0,         # cells are never wider than this, so a shapeless blob is still approximated
    open_px=0,              # grey opening radius (native px) on the footprint field: 1 removes 1-px specks and spurs
    chamfer_min_m=5.0,      # cut corners the field says are cut, from this size; 0 = never
    simplify_px=0.45,       # final simplification of building outlines, native px (below line_sep)
    hstep_m=6.0,            # split a building where its height steps by more than this; 0 = off
    split_necks=True,       # split a blob at narrow necks into separate buildings
    neck_ratio=0.55,        # a neck narrower than this times the smaller part's width splits
    axis="own",             # 'own' | 'street' | 'tile' | 'none'
    axis_conf=0.35,         # trust a building's own axis above this confidence
    axis_snap_deg=5.0,      # an own axis this close to the tile axis takes the tile axis
    # streets
    street_min_m2=25.0,     # specks of street smaller than this are removed, holes filled
    spur_m=20.0,            # dead-end stubs shorter than this are removed
    isolated_m=30.0,        # isolated bits of street shorter than this are removed
    merge_m=10.0,           # junctions closer than this are one junction
    loop_m=45.0,            # a second, longer connection between the same two junctions shorter than this is a skeleton loop
    face_m2=500.0,          # a loop of streets enclosing less than this is a skeleton artefact and collapses to a junction
    street_eps_px=1.4,      # Douglas-Peucker tolerance for the centrelines, native px
    street_snap_deg=8.0,    # runs this close to the grid axis are snapped to it
    street_snap45=False,    # also snap to the 45 degree family
    node_move_px=2.5,       # a junction never moves further than this when re-intersected
    width_min_m=5.0, width_max_m=32.0,
    # greens
    min_green_m2=40.0,
    green_eps_px=0.7,
    # blocks
    min_block_m2=60.0,
    street_overlap=0.2,     # a building overlapping a street plate by more than this share is clipped to the block
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _opts(kw):
    o = dict(DEFAULTS)
    o.update({k: v for k, v in kw.items() if v is not None})
    return o


def decode_fields(ch: np.ndarray, lut: HeightLUT | None, trunc: float = SDF_TRUNC_PX):
    """bytes [4,H,W] -> (footprint sdf, street sdf, green sdf) in native px, heights in metres (every pixel)"""
    f = lambda c: (c.astype(np.float32) - 128.0) / 127.0 * trunc  # noqa: E731
    if lut is not None:
        grey = np.asarray(lut.inverse, dtype=np.float64)[ch[1]]
    else:
        grey = ch[1].astype(np.float64)
    h = grey_to_metres(grey).astype(np.float32)
    return f(ch[0]), f(ch[2]), f(ch[3]), h


def _label(mask, eight=False):
    st = np.ones((3, 3), bool) if eight else None
    lab, n = ndimage.label(mask, structure=st)
    return lab, int(n)


def clean_mask(mask: np.ndarray, min_px: float, min_hole_px: float, eight=False) -> np.ndarray:
    """drop components smaller than min_px, fill enclosed holes smaller than min_hole_px"""
    m = mask.copy()
    if min_px > 0:
        lab, n = _label(m, eight)
        if n:
            sizes = np.bincount(lab.ravel())
            m &= sizes[lab] >= min_px
    if min_hole_px > 0:
        lab, n = _label(~m, not eight)
        if n:
            sizes = np.bincount(lab.ravel())
            border = np.zeros(n + 1, bool)
            for edge in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
                border[np.unique(edge)] = True
            small = (sizes < min_hole_px) & ~border
            small[0] = False
            m |= small[lab]
    return m


def _bilinear(arr, x, y):
    """sample arr (indexed [y, x], pixel centres at +0.5) at corner coordinates"""
    return ndimage.map_coordinates(arr, [np.asarray(y, float) - 0.5, np.asarray(x, float) - 0.5],
                                   order=1, mode="nearest")


def _polys(geom):
    """list of Polygons in any shapely geometry"""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out += _polys(g)
        return out
    return []


def _ring(r):
    pts = [[round(float(x), 3), round(float(y), 3)] for x, y in r.coords[:-1]]
    return pts


def _part(poly: Polygon, h=None):
    d = {"outer": _ring(poly.exterior), "holes": [_ring(i) for i in poly.interiors]}
    if h is not None:
        d["h"] = round(float(h), 2)
    return d


# ---------------------------------------------------------------------------
# dominant directions
# ---------------------------------------------------------------------------

def axis_vote(sdf: np.ndarray, where: np.ndarray | None = None, band: float = 1.5):
    """
    Axis (radians, in (-pi/4, pi/4]) and confidence of the edges in a distance field.
    Every pixel near the zero crossing votes with its gradient direction times four,
    so directions 90 degrees apart add up instead of cancelling.
    """
    gy, gx = np.gradient(sdf)
    g = np.hypot(gx, gy)
    sel = (np.abs(sdf) < band) & (g > 0.25)
    if where is not None:
        sel &= where
    if sel.sum() < 6:
        return 0.0, 0.0, int(sel.sum())
    a = 4.0 * np.arctan2(gy[sel], gx[sel])
    w = g[sel]
    C, S = float((w * np.cos(a)).sum()), float((w * np.sin(a)).sum())
    tot = float(w.sum())
    return math.atan2(S, C) / 4.0, math.hypot(C, S) / tot, int(sel.sum())


def _ang_diff90(a, b):
    d = a - b
    q = math.pi / 2
    return d - round(d / q) * q


# ---------------------------------------------------------------------------
# streets: skeleton -> graph -> clean -> straight runs -> plates
# ---------------------------------------------------------------------------

_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _skeleton_graph(skel: np.ndarray):
    """nodes: {id: [x, y]} (corner coords), edges: [{a, b, px: [(x, y), ...]}] with pixel centres"""
    H, W = skel.shape
    P = np.pad(skel.astype(np.uint8), 1)
    n8 = sum(P[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx] for dy, dx in _N8)
    junction = skel & (n8 >= 3)
    jl, nj = _label(junction, eight=True)
    nodes = {}
    for k, (cy, cx) in enumerate(ndimage.center_of_mass(junction, jl, range(1, nj + 1)), start=1):
        nodes[k] = [float(cx) + 0.5, float(cy) + 0.5]
    paths = skel & ~junction
    pl, npth = _label(paths, eight=True)
    edges = []
    next_id = nj + 1
    jl_pad = np.pad(jl, 1)
    for k in range(1, npth + 1):
        ys, xs = np.nonzero(pl == k)
        pts = set(zip(xs.tolist(), ys.tolist()))
        # ends: path pixels with at most one path neighbour
        def path_nb(p):
            return [(p[0] + dx, p[1] + dy) for dy, dx in _N8 if (p[0] + dx, p[1] + dy) in pts]
        ends = [p for p in pts if len(path_nb(p)) <= 1]
        if not ends:                       # a cycle: break it anywhere
            ends = [next(iter(pts))]
        start = ends[0]
        order = [start]
        seen = {start}
        cur = start
        while True:
            nb = [q for q in path_nb(cur) if q not in seen]
            if not nb:
                break
            # prefer 4-neighbours so diagonal shortcuts do not skip pixels
            nb.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            cur = nb[0]
            seen.add(cur)
            order.append(cur)
        if len(seen) < len(pts):           # a branching path (should not happen after junction removal); take what we walked
            pass

        def touching_junction(p):
            ids = set()
            for dy, dx in _N8:
                j = jl_pad[p[1] + 1 + dy, p[0] + 1 + dx]
                if j:
                    ids.add(int(j))
            return ids

        a_ids = touching_junction(order[0])
        b_ids = touching_junction(order[-1])
        if len(order) == 1 and a_ids and len(a_ids) >= 2:     # one pixel linking two junction clusters
            ids = sorted(a_ids)
            a, b = ids[0], ids[1]
        else:
            if a_ids:
                a = min(a_ids)
            else:
                a = next_id; nodes[a] = [order[0][0] + 0.5, order[0][1] + 0.5]; next_id += 1
            b_ids.discard(a) if len(order) > 1 else None
            if b_ids:
                b = min(b_ids)
            else:
                b = next_id; nodes[b] = [order[-1][0] + 0.5, order[-1][1] + 0.5]; next_id += 1
        edges.append({"a": a, "b": b, "px": [(x + 0.5, y + 0.5) for x, y in order]})
    # junction clusters that touch each other directly with no path pixel between: merge later via merge_m
    return nodes, edges


def _edge_pts(e, nodes):
    return [tuple(nodes[e["a"]])] + e["px"] + [tuple(nodes[e["b"]])]


def _length(pts):
    p = np.asarray(pts, float)
    if len(p) < 2:
        return 0.0
    return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())


def _degrees(nodes, edges):
    deg = {k: 0 for k in nodes}
    for e in edges:
        deg[e["a"]] += 1
        deg[e["b"]] += 1
    return deg


def _split_loops(nodes, edges, merge_px):
    """a path that leaves a junction and returns to it (two streets meeting again at the tile
    edge, or around a hole) becomes two edges joined at its farthest point"""
    out = []
    next_id = (max(nodes) + 1) if nodes else 1
    for e in edges:
        if e["a"] != e["b"] or len(e["px"]) < 3:
            out.append(e); continue
        p0 = np.asarray(nodes[e["a"]])
        px = np.asarray(e["px"])
        k = int(np.argmax(np.hypot(px[:, 0] - p0[0], px[:, 1] - p0[1])))
        if np.hypot(*(px[k] - p0)) < merge_px:
            continue                                    # a tiny loop at a junction: drop it
        nodes[next_id] = [float(px[k][0]), float(px[k][1])]
        out.append({"a": e["a"], "b": next_id, "px": e["px"][:k]})
        out.append({"a": next_id, "b": e["b"], "px": e["px"][k + 1:]})
        next_id += 1
    return nodes, out


def _clean_graph(nodes, edges, spur_px, isolated_px, merge_px):
    nodes, edges = _split_loops(nodes, edges, merge_px)
    for _ in range(3):
        changed = False
        deg = _degrees(nodes, edges)
        keep = []
        for e in edges:
            L = _length(_edge_pts(e, nodes))
            da, db = deg[e["a"]], deg[e["b"]]
            if da == 1 and db == 1 and L < isolated_px:
                changed = True; continue
            if (da == 1) != (db == 1) and L < spur_px:
                changed = True; continue
            if e["a"] == e["b"] and L < 2 * merge_px:          # tiny loop at a junction
                changed = True; continue
            keep.append(e)
        edges = keep
        # contract short junction-to-junction edges
        deg = _degrees(nodes, edges)
        for e in list(edges):
            if e["a"] == e["b"]:
                continue
            L = _length(_edge_pts(e, nodes))
            if L < merge_px and deg[e["a"]] >= 2 and deg[e["b"]] >= 2:
                a, b = e["a"], e["b"]
                pa, pb = nodes[a], nodes[b]
                nodes[a] = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2]
                edges.remove(e)
                for f in edges:
                    if f["a"] == b: f["a"] = a
                    if f["b"] == b: f["b"] = a
                del nodes[b]
                deg = _degrees(nodes, edges)
                changed = True
        # remove degree-2 nodes by joining their two edges
        deg = _degrees(nodes, edges)
        for n, d in list(deg.items()):
            if d != 2:
                continue
            inc = [e for e in edges if e["a"] == n or e["b"] == n]
            if len(inc) != 2 or inc[0] is inc[1]:
                continue
            e1, e2 = inc
            p1 = _edge_pts(e1, nodes); p2 = _edge_pts(e2, nodes)
            if e1["b"] != n: p1 = p1[::-1]; e1_far = e1["b"]
            else: e1_far = e1["a"]
            if e2["a"] != n: p2 = p2[::-1]; e2_far = e2["a"]
            else: e2_far = e2["b"]
            if e1_far == e2_far:
                continue                                       # joining them would make a loop; keep the node
            px = p1[1:-1] + [tuple(nodes[n])] + p2[1:-1]
            edges.remove(e1); edges.remove(e2)
            edges.append({"a": e1_far, "b": e2_far, "px": px})
            del nodes[n]
            deg = _degrees(nodes, edges)
            changed = True
        # drop orphan nodes
        used = set()
        for e in edges:
            used.add(e["a"]); used.add(e["b"])
        for n in list(nodes):
            if n not in used:
                del nodes[n]
        if not changed:
            break
    return nodes, edges


def _dp(pts: np.ndarray, eps: float):
    """Douglas-Peucker on an open polyline; returns indices kept"""
    n = len(pts)
    if n < 3:
        return list(range(n))
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        L = math.hypot(*d)
        seg = pts[i + 1:j]
        if L < 1e-9:
            dist = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            dist = np.abs(d[0] * (a[1] - seg[:, 1]) - d[1] * (a[0] - seg[:, 0])) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k)); stack.append((i + 1 + k, j))
    return [int(i) for i in np.nonzero(keep)[0]]


def _edge_width(dt, e, pts, px_m, o):
    """
    Width of a street in metres from the distance transform of the street mask along the
    edge's skeleton pixels. The mask, not the depth of the field, carries the width: a
    generated field can be shallow (byte 140 at the centre of a 3-pixel street) while the
    thresholded band is still the right width. A skeleton pixel sits up to half a pixel off
    the true centre of an even-width band, hence the -0.25 rather than -0.5.
    """
    px = np.asarray(e["px"], float) if len(e["px"]) >= 1 else np.asarray(pts, float)
    ix = np.clip(np.floor(px[:, 0]).astype(int), 0, dt.shape[1] - 1)
    iy = np.clip(np.floor(px[:, 1]).astype(int), 0, dt.shape[0] - 1)
    hw = float(np.mean(dt[iy, ix])) - 0.25
    return float(np.clip(2.0 * hw * px_m, o["width_min_m"], o["width_max_m"]))


def _drop_lenses(nodes, edges, loop_px):
    """two short edges between the same pair of junctions (a skeleton loop around a blob): keep the shorter"""
    seen = {}
    out = []
    for e in sorted(edges, key=lambda e: _length(_edge_pts(e, nodes))):
        key = (min(e["a"], e["b"]), max(e["a"], e["b"]))
        L = _length(_edge_pts(e, nodes))
        if key in seen and L < loop_px and e["a"] != e["b"]:
            continue
        seen[key] = True
        out.append(e)
    return out


def _collapse_small_faces(nodes, edges, area_px):
    """
    Loops the skeleton draws around a blob (a wide junction, a hole in a generated street)
    show up as small faces of the graph. Every node on a face smaller than area_px is merged
    into one node at the face's centre, and the edges that became loops are dropped.
    """
    from shapely.ops import polygonize
    if not edges:
        return nodes, edges
    lines = [LineString(_edge_pts(e, nodes)) for e in edges if len(_edge_pts(e, nodes)) >= 2]
    try:
        faces = [f for f in polygonize(unary_union(lines)) if f.area < area_px]
    except Exception:
        return nodes, edges
    if not faces:
        return nodes, edges
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    centres = {}
    for f in faces:
        ring = f.exterior
        on = [n for n, p in nodes.items() if ring.distance(shapely.geometry.Point(p)) < 0.05]
        if len(on) < 2:
            continue
        root = find(on[0])
        for n in on[1:]:
            parent[find(n)] = root
        centres[root] = (f.centroid.x, f.centroid.y)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    for root, members in groups.items():
        if len(members) < 2:
            continue
        cx, cy = centres.get(root, (np.mean([nodes[m][0] for m in members]), np.mean([nodes[m][1] for m in members])))
        nodes[root] = [float(cx), float(cy)]
        for e in edges:
            if e["a"] in members: e["a"] = root
            if e["b"] in members: e["b"] = root
        for m in members:
            if m != root:
                del nodes[m]
    edges = [e for e in edges if e["a"] != e["b"]]
    # a merged node no longer sits on its old pixel paths: trim the path pixels that now lie behind it
    for e in edges:
        for end in ("a", "b"):
            p = np.asarray(nodes[e[end]])
            while len(e["px"]) > 1:
                q = e["px"][0] if end == "a" else e["px"][-1]
                r = e["px"][1] if end == "a" else e["px"][-2]
                if np.hypot(*(np.asarray(q) - p)) > np.hypot(*(np.asarray(r) - p)) + 0.3:
                    e["px"] = e["px"][1:] if end == "a" else e["px"][:-1]
                else:
                    break
    return nodes, edges


def streets_from_field(sdf_st: np.ndarray, o: dict, px_m: float):
    """
    -> dict(nodes, runs, edges, plates, axis)
       runs:  [{a, b, w}] straight segments between node ids (a, b), width in metres
       edges: [{pts: [[x, y], ...], w}] the cleaned centrelines, one per street
    """
    H, W = sdf_st.shape
    area = lambda m2: m2 / (px_m * px_m)  # noqa: E731
    mask = clean_mask(sdf_st > 0, area(o["street_min_m2"]), area(o["street_min_m2"]), eight=True)
    empty = dict(nodes={}, runs=[], edges=[], plates=Polygon(), axis={"deg": 0.0, "conf": 0.0})
    if not mask.any():
        return empty
    skel = skeletonize(mask)
    dt = ndimage.distance_transform_edt(mask)
    nodes, edges = _skeleton_graph(skel)
    # a stub is an artefact of the skeleton when it is not longer than the street is wide
    wpx = float(np.mean(dt[skel])) * 2 if skel.any() else 2.0
    spur_px = max(o["spur_m"] / px_m, 1.5 * wpx + 2.0)
    nodes, edges = _clean_graph(nodes, edges, spur_px, o["isolated_m"] / px_m, o["merge_m"] / px_m)
    edges = _drop_lenses(nodes, edges, o["loop_m"] / px_m)
    nodes, edges = _collapse_small_faces(nodes, edges, o["face_m2"] / (px_m * px_m))
    nodes, edges = _clean_graph(nodes, edges, spur_px, o["isolated_m"] / px_m, o["merge_m"] / px_m)
    if not edges:
        return empty

    # straight runs: DP on every edge, interior vertices become degree-2 nodes
    runs = []       # dicts: a, b, d (unit dir), c (centroid), L, w (metres)
    next_id = max(nodes) + 1
    edge_nodes = []  # per edge: list of node ids along it
    for e in edges:
        pts = np.asarray(_edge_pts(e, nodes), float)
        keep = _dp(pts, o["street_eps_px"])
        ids = [e["a"]]
        for k in keep[1:-1]:
            nodes[next_id] = [float(pts[k, 0]), float(pts[k, 1])]
            ids.append(next_id); next_id += 1
        ids.append(e["b"])
        w = round(_edge_width(dt, e, pts, px_m, o) * 2) / 2
        for i in range(len(ids) - 1):
            seg = pts[keep[i]:keep[i + 1] + 1]
            c = seg.mean(0)
            if len(seg) >= 3:
                u, s, vt = np.linalg.svd(seg - c, full_matrices=False)
                d = vt[0]
            else:
                d = seg[-1] - seg[0]
            n = math.hypot(*d)
            d = d / n if n > 1e-9 else np.array([1.0, 0.0])
            if np.dot(d, seg[-1] - seg[0]) < 0:
                d = -d
            runs.append({"a": ids[i], "b": ids[i + 1], "d": d, "c": c, "L": float(_length(seg)), "w": w})
        edge_nodes.append({"ids": ids, "w": w})

    # dominant street axis (4-theta vote weighted by length) and snapping
    C = sum(r["L"] * math.cos(4 * math.atan2(r["d"][1], r["d"][0])) for r in runs)
    S = sum(r["L"] * math.sin(4 * math.atan2(r["d"][1], r["d"][0])) for r in runs)
    tot = sum(r["L"] for r in runs) or 1.0
    theta0 = math.atan2(S, C) / 4.0
    conf = math.hypot(C, S) / tot
    snap = math.radians(o["street_snap_deg"])
    for r in runs:
        if r["L"] < 2.0:
            continue
        ang = math.atan2(r["d"][1], r["d"][0])
        fams = [math.pi / 2]
        if o["street_snap45"]:
            fams.append(math.pi / 4)
        for q in fams:
            d = (ang - theta0) - round((ang - theta0) / q) * q
            if abs(d) < snap:
                a2 = ang - d
                r["d"] = np.array([math.cos(a2), math.sin(a2)])
                break

    # re-intersect: every node to the least-squares point of its incident run lines
    deg = {}
    inc = {}
    for r in runs:
        for n in (r["a"], r["b"]):
            deg[n] = deg.get(n, 0) + 1
            inc.setdefault(n, []).append(r)
    lam_scale = 0.08
    for n, rs in inc.items():
        p0 = np.asarray(nodes[n], float)
        A = np.zeros((2, 2)); b = np.zeros(2); wsum = 0.0
        for r in rs:
            nrm = np.array([-r["d"][1], r["d"][0]])
            w = min(r["L"], 20.0)
            A += w * np.outer(nrm, nrm); b += w * nrm * float(np.dot(nrm, r["c"])); wsum += w
        lam = lam_scale * wsum + 1e-6
        A += lam * np.eye(2); b += lam * p0
        p = np.linalg.solve(A, b)
        mv = p - p0
        m = math.hypot(*mv)
        if m > o["node_move_px"]:
            p = p0 + mv * (o["node_move_px"] / m)
        nodes[n] = [float(p[0]), float(p[1])]

    # merge collinear runs at degree-2 nodes (they were DP vertices on a nearly straight street)
    for en in edge_nodes:
        ids = en["ids"]
        out = [ids[0]]
        for i in range(1, len(ids) - 1):
            p_prev, p, p_next = (np.asarray(nodes[out[-1]]), np.asarray(nodes[ids[i]]), np.asarray(nodes[ids[i + 1]]))
            d1 = p - p_prev; d2 = p_next - p
            n1, n2 = math.hypot(*d1), math.hypot(*d2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cosang = float(np.dot(d1, d2)) / (n1 * n2)
            if cosang > math.cos(math.radians(2.0)):
                continue
            out.append(ids[i])
        out.append(ids[-1])
        en["ids"] = out

    # final runs and centrelines from the node positions
    runs_out = []
    lines = []
    deg = {}
    for en in edge_nodes:
        for i in range(len(en["ids"]) - 1):
            a, b = en["ids"][i], en["ids"][i + 1]
            deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    # a street that dies within two pixels of the tile edge continues beyond it: carry its end
    # onto the edge along its own direction, so the plate reaches the border and cuts the block
    at_border = set()
    for en in edge_nodes:
        ids = en["ids"]
        for end, nxt in ((ids[0], ids[1]), (ids[-1], ids[-2])):
            if deg.get(end, 0) != 1:
                continue
            p = np.asarray(nodes[end], float); q = np.asarray(nodes[nxt], float)
            d = p - q; L = math.hypot(*d)
            if L < 1e-6:
                continue
            d /= L
            best = None
            tol = max(2.0, 0.8 * en["w"] / px_m)     # the skeleton stops about half a width short of the edge
            for axis, lim in ((0, 0.0), (0, float(W)), (1, 0.0), (1, float(H))):
                if abs(p[axis] - lim) > tol or abs(d[axis]) < 0.2:
                    continue
                t = (lim - p[axis]) / d[axis]
                if t < -1.0:
                    continue
                cand = p + d * t
                if -0.01 <= cand[0] <= W + 0.01 and -0.01 <= cand[1] <= H + 0.01 and (best is None or abs(t) < best[0]):
                    best = (abs(t), cand)
            if best is not None:
                nodes[end] = [float(best[1][0]), float(best[1][1])]
                at_border.add(end)
    for en in edge_nodes:
        ids = en["ids"]
        pts = [[round(nodes[i][0], 3), round(nodes[i][1], 3)] for i in ids]
        lines.append({"pts": pts, "w": en["w"]})
        for i in range(len(ids) - 1):
            runs_out.append({"a": ids[i], "b": ids[i + 1], "w": en["w"]})

    # plates: each run buffered by half its width, extended by half a width into junctions
    tile = box(0, 0, W, H)
    geoms = []
    for r in runs_out:
        pa, pb = np.asarray(nodes[r["a"]]), np.asarray(nodes[r["b"]])
        d = pb - pa; L = math.hypot(*d)
        if L < 1e-6:
            continue
        d = d / L
        hw = r["w"] / px_m / 2.0
        qa = pa - d * hw if (deg.get(r["a"], 1) >= 2 or r["a"] in at_border) else pa
        qb = pb + d * hw if (deg.get(r["b"], 1) >= 2 or r["b"] in at_border) else pb
        geoms.append(LineString([qa, qb]).buffer(hw, cap_style="flat", join_style="mitre", mitre_limit=2.0))
    plates = unary_union(geoms).intersection(tile) if geoms else Polygon()
    return dict(nodes={k: [round(v[0], 3), round(v[1], 3)] for k, v in nodes.items()},
                runs=runs_out, edges=lines, plates=plates,
                axis={"deg": math.degrees(theta0), "conf": conf})


# ---------------------------------------------------------------------------
# buildings
# ---------------------------------------------------------------------------

def _watershed(dt: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """priority flood from markers downhill on dt (higher first); small, pure python"""
    lab = markers.copy()
    H, W = dt.shape
    heap = []
    ys, xs = np.nonzero(markers)
    for y, x in zip(ys.tolist(), xs.tolist()):
        heapq.heappush(heap, (-float(dt[y, x]), y, x))
    while heap:
        _, y, x = heapq.heappop(heap)
        l = lab[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and lab[yy, xx] == 0:
                lab[yy, xx] = l
                heapq.heappush(heap, (-float(dt[yy, xx]), yy, xx))
    return lab


def split_necks(mask: np.ndarray, ratio: float, min_px: float):
    """split a component mask into parts joined by necks narrower than ratio * the parts' widths"""
    dt = ndimage.distance_transform_edt(mask)
    mx = ndimage.maximum_filter(dt, size=5)
    peaks = mask & (dt >= mx - 0.3) & (dt >= 1.2)
    ml, nm = _label(peaks, eight=True)
    if nm <= 1:
        return mask.astype(np.int32)
    lab = _watershed(dt, ml, mask)
    # peak height per region
    ph = ndimage.maximum(dt, lab, range(1, nm + 1))
    ph = np.concatenate([[0.0], np.asarray(ph)])
    # neck between neighbouring regions: the max dt along their shared boundary
    parent = list(range(nm + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    necks = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = lab[:-dy or None, :-dx or None] if (dy or dx) else lab
        b = lab[dy:, dx:]
        da = dt[:-dy or None, :-dx or None]
        db = dt[dy:, dx:]
        sel = (a > 0) & (b > 0) & (a != b)
        for i, j, v in zip(a[sel].tolist(), b[sel].tolist(), np.maximum(da[sel], db[sel]).tolist()):
            key = (min(i, j), max(i, j))
            if v > necks.get(key, 0.0):
                necks[key] = v
    for (i, j), v in sorted(necks.items(), key=lambda kv: -kv[1]):
        if v >= ratio * min(ph[i], ph[j]):
            parent[find(i)] = find(j)
    out = np.zeros_like(lab)
    for k in range(1, nm + 1):
        out[lab == k] = find(k)
    # parts too small go to their largest neighbour
    ids, counts = np.unique(out[out > 0], return_counts=True)
    small = [i for i, c in zip(ids, counts) if c < min_px]
    for i in small:
        m = out == i
        ring = ndimage.binary_dilation(m, iterations=1) & mask & ~m
        if ring.any():
            nb, cnt = np.unique(out[ring], return_counts=True)
            out[m] = nb[np.argmax(cnt)]
    # relabel 1..k
    ids = np.unique(out[out > 0])
    rl = np.zeros(out.max() + 1, np.int32)
    rl[ids] = np.arange(1, len(ids) + 1)
    return rl[out]


def split_heights(piece: np.ndarray, h: np.ndarray, step_m: float, min_px: float):
    """plateau labels (1..k) inside piece from the height map; 1 label when it is flat"""
    if step_m <= 0:
        return piece.astype(np.int32)
    ys, xs = np.nonzero(piece)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    loc = piece[y0:y1, x0:x1]
    hl = h[y0:y1, x0:x1]
    bins = np.where(loc, np.rint(hl / step_m).astype(np.int32), -999)
    vals = np.unique(bins[loc])
    if len(vals) == 1:
        return piece.astype(np.int32)
    k3 = np.array([[1, 1, 1], [1, 1.5, 1], [1, 1, 1]], np.float32)
    for _ in range(2):
        best = np.full(bins.shape, -999, np.int32); bc = np.zeros(bins.shape, np.float32)
        for v in vals:
            c = ndimage.convolve((bins == v).astype(np.float32), k3, mode="constant")
            better = loc & (c > bc)
            best[better] = v; bc[better] = c[better]
        bins = np.where(loc, best, -999)
    # regions of equal bin; absorb small ones into the neighbour they touch most
    for _ in range(6):
        lab = np.zeros(bins.shape, np.int32); n = 0
        for v in np.unique(bins[loc]):
            l, k = _label(bins == v)
            lab[l > 0] = l[l > 0] + n; n += k
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = [i for i in range(1, n + 1) if sizes[i] < min_px]
        if not small or n == 1:
            break
        changed = False
        for i in small:
            m = lab == i
            ring = ndimage.binary_dilation(m) & loc & ~m
            if ring.any():
                nb, cnt = np.unique(bins[ring], return_counts=True)
                bins[m] = nb[np.argmax(cnt)]; changed = True
        if not changed:
            break
    out = np.zeros(piece.shape, np.int32)
    out[y0:y1, x0:x1] = lab
    return out


def _peaks_1d(pos, w, nbins, sep, min_votes):
    """weighted 1-D histogram peaks with a minimum separation; returns refined positions"""
    if len(pos) == 0:
        return []
    hist = np.bincount(np.clip(pos.astype(int), 0, nbins - 1), weights=w, minlength=nbins).astype(np.float64)
    sm = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    order = np.argsort(-sm)
    picked = []
    for i in order:
        if sm[i] < min_votes:
            break
        if all(abs(i - p) >= sep for p in picked):
            picked.append(int(i))
    out = []
    for p in picked:
        sel = np.abs(pos - (p + 0.5)) <= sep / 2
        if sel.any():
            out.append(float(np.average(pos[sel], weights=w[sel])))
        else:
            out.append(p + 0.5)
    return sorted(out)


def _complete_lines(lines, lo, hi, sep, max_gap, size):
    out = sorted(lines)
    for v in (lo, hi):
        if all(abs(v - p) >= sep * 0.75 for p in out):
            out.append(v)
    out = sorted(out)
    full = [0.0]
    for i, v in enumerate(out):
        prev = full[-1]
        gap = v - prev
        if i > 0 and gap > max_gap:
            k = int(math.ceil(gap / max_gap))
            full += [prev + gap * j / k for j in range(1, k)]
        full.append(v)
    if size - full[-1] > 0.5:
        full.append(float(size))
    return full


def fit_piece(sdf: np.ndarray, plateau: np.ndarray, h: np.ndarray, theta: float, o: dict, px_m: float):
    """
    One building piece (plateau > 0 marks its pixels) -> list of (Polygon in native coords, height m).
    The polygon edges lie on the lines detected in the piece's own frame.
    """
    up = int(o["up"])
    ys, xs = np.nonzero(plateau > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
    bw, bh = x1 + 1 - x0, y1 + 1 - y0
    c, s = math.cos(theta), math.sin(theta)
    margin = 2.0
    hw = (bw / 2) * abs(c) + (bh / 2) * abs(s) + margin
    hh = (bw / 2) * abs(s) + (bh / 2) * abs(c) + margin
    Fw, Fh = int(math.ceil(2 * hw * up)), int(math.ceil(2 * hh * up))
    u = (np.arange(Fw) + 0.5) / up - hw
    v = (np.arange(Fh) + 0.5) / up - hh
    U, V = np.meshgrid(u, v)
    NX = cx + c * U - s * V
    NY = cy + s * U + c * V
    F = ndimage.map_coordinates(sdf, [NY - 0.5, NX - 0.5], order=1, mode="nearest")
    P = ndimage.map_coordinates(plateau, [NY - 0.5, NX - 0.5], order=0, mode="constant", cval=0)
    dil = ndimage.binary_dilation(plateau > 0, iterations=1)
    D = ndimage.map_coordinates(dil.astype(np.uint8), [NY - 0.5, NX - 0.5], order=0, mode="constant", cval=0) > 0
    F = np.where(D, F, -SDF_TRUNC_PX)
    inside = F > 0
    if not inside.any():
        return []

    # boundary samples vote for the lines they lie on
    gy, gx = np.gradient(F)
    g = np.hypot(gx, gy)
    band = (np.abs(F) < o["band_px"]) & (g > 1e-4) & D
    by, bx = np.nonzero(band)
    f = F[by, bx]; ggx = gx[by, bx] / g[by, bx]; ggy = gy[by, bx] / g[by, bx]
    # zero crossing of each sample: step against the gradient by the field value (native px -> frame px)
    zx = bx + 0.5 - f * up * ggx
    zy = by + 0.5 - f * up * ggy
    cos_lim = math.cos(math.radians(o["line_ang_deg"]))
    vert = np.abs(ggx) >= cos_lim      # edge runs along y: votes for a line x = const
    horz = np.abs(ggy) >= cos_lim
    sep = o["line_sep_m"] / px_m * up
    per_px = 2 * o["band_px"] * up * up          # votes one native px of straight edge produces
    min_votes = 0.45 * per_px * (o["line_min_m"] / px_m)
    xl = _peaks_1d(zx[vert], np.ones(vert.sum()), Fw, sep, min_votes)
    yl = _peaks_1d(zy[horz], np.ones(horz.sum()), Fh, sep, min_votes)
    # the extents of the piece are lines too (a piece with no straight edge still gets its box),
    # and long gaps are subdivided so an irregular blob is approximated at max_gap granularity
    # rather than dropped; the subdivision is invisible on clean shapes because the union of
    # the cells merges them again
    iy, ix = np.nonzero(inside)
    max_gap = o["max_gap_m"] / px_m * up
    xs_ = _complete_lines(xl, float(ix.min()), float(ix.max() + 1), sep, max_gap, Fw)
    ys_ = _complete_lines(yl, float(iy.min()), float(iy.max() + 1), sep, max_gap, Fh)

    # cells: inside by majority of the field, plateau by majority of the labels (integral images)
    ks = [int(k) for k in np.unique(P[inside])] or [1]
    xi = np.clip(np.rint(np.asarray(xs_)).astype(int), 0, Fw)
    yi = np.clip(np.rint(np.asarray(ys_)).astype(int), 0, Fh)

    def cellsums(img):
        I = np.pad(np.cumsum(np.cumsum(img.astype(np.float64), 0), 1), ((1, 0), (1, 0)))
        return I[yi[1:, None], xi[None, 1:]] - I[yi[:-1, None], xi[None, 1:]] - I[yi[1:, None], xi[None, :-1]] + I[yi[:-1, None], xi[None, :-1]]

    n_cell = (yi[1:] - yi[:-1])[:, None] * (xi[1:] - xi[:-1])[None, :]
    tot = cellsums(inside)
    ins = np.where(n_cell > 0, tot >= 0.5 * np.maximum(n_cell, 1), False)
    thin = n_cell == 0                       # thinner than a sample: point test at the centre
    if thin.any():
        cy_, cx_ = np.nonzero(thin)
        fx = (np.asarray(xs_)[cx_] + np.asarray(xs_)[cx_ + 1]) / 2; fy = (np.asarray(ys_)[cy_] + np.asarray(ys_)[cy_ + 1]) / 2
        ins[thin] = ndimage.map_coordinates(F, [fy - 0.5, fx - 0.5], order=1, mode="nearest") > 0
    if len(ks) == 1:
        lab = np.full(ins.shape, ks[0])
    else:
        stack = np.stack([cellsums(inside & (P == k)) for k in ks])
        lab = np.asarray(ks)[np.argmax(stack, 0)]
        if thin.any():
            cy_, cx_ = np.nonzero(thin)
            pk = ndimage.map_coordinates(P, [fy - 0.5, fx - 0.5], order=0, mode="nearest")
            lab[thin] = np.where(np.isin(pk, ks), pk, ks[0])
    cells = {k: [] for k in ks}
    for j, i in zip(*np.nonzero(ins)):
        cells[int(lab[j, i])].append(box(xs_[i], ys_[j], xs_[i + 1], ys_[j + 1]))

    out = []
    min_area_f = o["min_building_m2"] / (px_m * px_m) * up * up
    min_hole_f = o["min_hole_m2"] / (px_m * px_m) * up * up
    chamfer_min = o["chamfer_min_m"] / px_m * up
    for k, bxs in cells.items():
        if not bxs:
            continue
        geom = unary_union(bxs)
        for poly in _polys(geom):
            if poly.area < min_area_f:
                continue
            poly = Polygon(poly.exterior, [r for r in poly.interiors if Polygon(r).area >= min_hole_f])
            poly = shapely.simplify(poly, 0.01)
            if o["chamfer_min_m"] > 0:
                poly = _chamfer(poly, F, chamfer_min)
            # a wall that is a degree or two off the frame comes out of the cells as a staircase of
            # sub-pixel steps; simplifying below the line separation turns it back into one wall
            # (slightly off-axis, which is the truth) and cannot touch a real jog, which is a line apart
            poly = shapely.simplify(poly, o["simplify_px"] * up)
            # frame -> native
            def back(xy):
                a = np.asarray(xy, float)
                uu = a[:, 0] / up - hw; vv = a[:, 1] / up - hh
                return np.stack([cx + c * uu - s * vv, cy + s * uu + c * vv], 1)
            ext = back(poly.exterior.coords)
            ints = [back(r.coords) for r in poly.interiors]
            pn = Polygon(ext, ints)
            if not pn.is_valid:
                pn = pn.buffer(0)
            hk = float(np.median(h[plateau == k])) if (plateau == k).any() else float(np.median(h[plateau > 0]))
            for q in _polys(pn):
                out.append((q, hk))
    return out


def _chamfer(poly: Polygon, F: np.ndarray, min_size: float):
    """cut convex corners the field says are cut; sizes in frame px"""
    ext = np.asarray(poly.exterior.coords[:-1], float)
    n = len(ext)
    if n < 4:
        return poly
    area = 0.0
    for i in range(n):
        x0, y0 = ext[i]; x1, y1 = ext[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    sgn = 1.0 if area > 0 else -1.0
    vals = ndimage.map_coordinates(F, [ext[:, 1] - 0.5, ext[:, 0] - 0.5], order=1, mode="nearest")
    out = []
    changed = False
    for i in range(n):
        p = ext[i]; q = ext[(i - 1) % n]; r = ext[(i + 1) % n]
        d1 = q - p; d2 = r - p
        l1, l2 = math.hypot(*d1), math.hypot(*d2)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        convex = (cross * sgn) < 0
        v = float(vals[i])
        if not convex or v > -0.8 or l1 < 1e-6 or l2 < 1e-6:
            out.append(p); continue
        a = -v * math.sqrt(2.0)                  # leg of the cut triangle
        if a < min_size or a > 0.4 * min(l1, l2):
            out.append(p); continue
        p1 = p + d1 / l1 * a; p2 = p + d2 / l2 * a
        m = (p1 + p2) / 2
        vm = float(ndimage.map_coordinates(F, [[m[1] - 0.5], [m[0] - 0.5]], order=1, mode="nearest")[0])
        v1 = float(ndimage.map_coordinates(F, [[p1[1] - 0.5], [p1[0] - 0.5]], order=1, mode="nearest")[0])
        v2 = float(ndimage.map_coordinates(F, [[p2[1] - 0.5], [p2[0] - 0.5]], order=1, mode="nearest")[0])
        if abs(vm) > 0.6 or abs(v1) > 0.9 or abs(v2) > 0.9:   # a real chamfer is a straight cut, not a rounded blob
            out.append(p); continue
        out.append(p1); out.append(p2); changed = True
    if not changed:
        return poly
    res = Polygon(out, list(poly.interiors))
    return res if res.is_valid else poly


def buildings_from_field(sdf_fp: np.ndarray, h: np.ndarray, o: dict, px_m: float, tile_axis: float,
                         street_runs=None, street_nodes=None):
    H, W = sdf_fp.shape
    a = lambda m2: m2 / (px_m * px_m)  # noqa: E731
    mask = clean_mask(sdf_fp > 0, a(o["min_building_m2"]), a(o["min_hole_m2"]))
    lab, n = _label(mask)
    out = []
    stats = {"components": n, "pieces": 0, "own_axis": 0, "street_axis": 0, "tile_axis": 0}
    # street run segments for the axis fallback
    segs = []
    if street_runs and street_nodes:
        for r in street_runs:
            pa, pb = np.asarray(street_nodes[r["a"]]), np.asarray(street_nodes[r["b"]])
            if np.hypot(*(pb - pa)) > 1e-6:
                segs.append((pa, pb))
    for k in range(1, n + 1):
        comp = lab == k
        if o["split_necks"]:
            parts = split_necks(comp, o["neck_ratio"], a(o["min_building_m2"]))
        else:
            parts = comp.astype(np.int32)
        nparts = int(parts.max())
        for pi in range(1, nparts + 1):
            piece = parts == pi
            if piece.sum() < a(o["min_building_m2"]):
                continue
            stats["pieces"] += 1
            # the field this piece is fitted on
            if nparts > 1:
                own = ndimage.distance_transform_edt(piece) - 0.5
                own_out = -(ndimage.distance_transform_edt(~piece) - 0.5)
                psdf = np.minimum(sdf_fp, np.where(piece, own, own_out).astype(np.float32))
            else:
                psdf = sdf_fp
            # axis
            theta = tile_axis
            mode = o["axis"]
            if mode == "none":
                theta = 0.0
            elif mode in ("own", "street"):
                got = False
                if mode == "own":
                    dil = ndimage.binary_dilation(piece, iterations=2)
                    th, conf, cnt = axis_vote(psdf, dil, 1.5)
                    if conf >= o["axis_conf"] and cnt >= 12:
                        theta = th; got = True; stats["own_axis"] += 1
                if not got and segs:
                    ys, xs = np.nonzero(piece)
                    cpt = np.array([xs.mean() + 0.5, ys.mean() + 0.5])
                    best, bd = None, 1e9
                    for pa, pb in segs:
                        d = pb - pa; L2 = float(np.dot(d, d))
                        t = float(np.clip(np.dot(cpt - pa, d) / L2, 0, 1))
                        dist = float(np.hypot(*(pa + t * d - cpt)))
                        if dist < bd:
                            bd, best = dist, d
                    if best is not None and bd < 15.0 / px_m * 3.125:
                        theta = math.atan2(best[1], best[0]); got = True; stats["street_axis"] += 1
                if not got:
                    stats["tile_axis"] += 1
                # normalise to (-45, 45] and snap to the tile axis when close
                theta = _ang_diff90(theta, 0.0)
                if abs(_ang_diff90(theta, tile_axis)) < math.radians(o["axis_snap_deg"]):
                    theta = tile_axis
            else:
                stats["tile_axis"] += 1
            plateau = split_heights(piece, h, o["hstep_m"], a(o["min_building_m2"]))
            out += fit_piece(psdf, plateau, h, theta, o, px_m)
    return out, stats


# ---------------------------------------------------------------------------
# greens
# ---------------------------------------------------------------------------

def greens_from_field(sdf_gr: np.ndarray, o: dict, px_m: float):
    H, W = sdf_gr.shape
    a = lambda m2: m2 / (px_m * px_m)  # noqa: E731
    mask = clean_mask(sdf_gr > 0, a(o["min_green_m2"]), a(o["min_hole_m2"]))
    if not mask.any():
        return Polygon()
    up = 2
    NY, NX = np.mgrid[0:H * up, 0:W * up]
    NX = (NX + 0.5) / up; NY = (NY + 0.5) / up
    F = ndimage.map_coordinates(sdf_gr, [NY - 0.5, NX - 0.5], order=1, mode="nearest")
    D = ndimage.map_coordinates(ndimage.binary_dilation(mask).astype(np.uint8), [NY - 0.5, NX - 0.5], order=0) > 0
    inside = (F > 0) & D
    ys, xs = np.nonzero(inside)
    if len(xs) == 0:
        return Polygon()
    # union of rows of runs (far fewer boxes than pixels)
    bxs = []
    for y in np.unique(ys):
        row = np.sort(xs[ys == y])
        breaks = np.nonzero(np.diff(row) > 1)[0]
        starts = np.concatenate([[0], breaks + 1]); ends = np.concatenate([breaks, [len(row) - 1]])
        for s0, e0 in zip(starts, ends):
            bxs.append(box(row[s0] / up, y / up, (row[e0] + 1) / up, (y + 1) / up))
    g = unary_union(bxs)
    g = shapely.simplify(g, o["green_eps_px"])
    g = g.buffer(0)
    keep = [p for p in _polys(g) if p.area >= a(o["min_green_m2"])]
    return unary_union(keep) if keep else Polygon()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def vectorize(ch: np.ndarray, lut: HeightLUT | None, tile_m: float = 400.0, **kw) -> dict:
    o = _opts(kw)
    C, H, W = ch.shape
    px_m = tile_m / W
    fp, st, gr, h = decode_fields(ch, lut)
    if o["smooth_px"] > 0:
        fp = ndimage.gaussian_filter(fp, o["smooth_px"]); st = ndimage.gaussian_filter(st, o["smooth_px"])
        gr = ndimage.gaussian_filter(gr, o["smooth_px"])
    if o["open_px"] > 0:
        # grey opening: removes specks and spurs thinner than the window, leaves straight edges where they are
        k = 2 * int(o["open_px"]) + 1
        fp = ndimage.maximum_filter(ndimage.minimum_filter(fp, size=k), size=k)
    tile = box(0, 0, W, H)

    # tile axis from streets and buildings
    th_st, c_st, n_st = axis_vote(st)
    th_fp, c_fp, n_fp = axis_vote(fp)
    if n_st + n_fp:
        C = c_st * n_st * math.cos(4 * th_st) + c_fp * n_fp * math.cos(4 * th_fp)
        S = c_st * n_st * math.sin(4 * th_st) + c_fp * n_fp * math.sin(4 * th_fp)
        tile_axis = math.atan2(S, C) / 4.0
        tile_conf = math.hypot(C, S) / (n_st + n_fp)
    else:
        tile_axis, tile_conf = 0.0, 0.0

    streets = streets_from_field(st, o, px_m)
    plates = streets["plates"]
    blds, bstats = buildings_from_field(fp, h, o, px_m, tile_axis, streets["runs"], streets["nodes"])
    green = greens_from_field(gr, o, px_m)

    # assembly: buildings and greens off the streets, greens off the buildings, blocks = tile - plates
    a = lambda m2: m2 / (px_m * px_m)  # noqa: E731
    bld_out = []
    for poly, hk in blds:
        q = poly.intersection(tile)
        if not plates.is_empty and q.intersects(plates):
            # a building that merely touches a street keeps its shape (the plate is under it);
            # one the raster drew well into the street is cut back to the block
            inter = q.intersection(plates).area
            if inter > o["street_overlap"] * q.area:
                q = q.difference(plates)
        for p in _polys(q):
            if p.area >= a(o["min_building_m2"]):
                bld_out.append((p, hk))
    bld_union = unary_union([p for p, _ in bld_out]) if bld_out else Polygon()
    green_out = []
    if not green.is_empty:
        g = green.difference(plates) if not plates.is_empty else green
        g = g.difference(bld_union) if not bld_union.is_empty else g
        green_out = [p for p in _polys(g) if p.area >= a(o["min_green_m2"])]
    blocks = [p for p in _polys(tile.difference(plates)) if p.area >= a(o["min_block_m2"])] if not plates.is_empty else [tile]

    res = {
        "W": W, "H": H, "tile_m": tile_m, "px_m": px_m,
        "axis": {"deg": math.degrees(tile_axis), "conf": tile_conf},
        "street_axis": streets["axis"],
        "buildings": [_part(p, hk) for p, hk in bld_out],
        "blocks": [_part(p) for p in blocks],
        "greens": [_part(p) for p in green_out],
        "plates": [_part(p) for p in _polys(plates)],
        "streets": streets["edges"],
        "stats": {**bstats, "buildings": len(bld_out), "streets": len(streets["edges"]),
                  "blocks": len(blocks), "greens": len(green_out)},
    }
    return res


# ---------------------------------------------------------------------------
# round trip: rasterise the result and compare with the tile it came from
# ---------------------------------------------------------------------------

def rasterize(res: dict, lut: HeightLUT | None = None, up: int = 4):
    """-> uint8 [4, H, W] in the mask encoding (footprint, height byte, street, green)"""
    from PIL import Image, ImageDraw
    W, H = res["W"], res["H"]

    def draw_parts(parts, value_of=None):
        im = Image.new("F", (W * up, H * up), 0.0)
        dr = ImageDraw.Draw(im)
        for p in parts:
            val = value_of(p) if value_of else 1.0
            dr.polygon([(x * up, y * up) for x, y in p["outer"]], fill=val)
            for hole in p["holes"]:
                dr.polygon([(x * up, y * up) for x, y in hole], fill=0.0)
        a = np.asarray(im, np.float32).reshape(H, up, W, up)
        return a

    fp = draw_parts(res["buildings"]).mean((1, 3)) >= 0.5
    st = draw_parts(res["plates"]).mean((1, 3)) >= 0.5
    gr = draw_parts(res["greens"]).mean((1, 3)) >= 0.5
    hm = draw_parts(res["buildings"], lambda p: p["h"])
    hmax = hm.max((1, 3))          # a pixel on a step takes the taller part
    height = np.zeros((H, W), np.uint8)
    if lut is not None:
        fwd = np.asarray(lut.forward)
        present = np.nonzero(fwd > 0)[0]
        grey = metres_to_grey(hmax).astype(int)
        nearest = present[np.abs(present[None, :] - grey.reshape(-1, 1)).argmin(1)].reshape(H, W)
        height = np.where(fp, fwd[nearest], 0).astype(np.uint8)
    else:
        height = np.where(fp, metres_to_grey(hmax), 0).astype(np.uint8)
    return np.stack([fp.astype(np.uint8) * 255, height, st.astype(np.uint8) * 255, gr.astype(np.uint8) * 255])


def roundtrip(ch: np.ndarray, res: dict, lut: HeightLUT | None = None, tile_m: float = 400.0) -> dict:
    from tile_codec import tile_metrics
    r = rasterize(res, lut)
    out = {}
    for name, i in (("footprint", 0), ("street", 2), ("green", 3)):
        a = ch[i] >= 128; b = r[i] >= 128
        inter = (a & b).sum(); union = (a | b).sum()
        out["iou_" + name] = float(inter / union) if union else 1.0
    if lut is not None:
        m0 = tile_metrics(ch, lut, tile_m); m1 = tile_metrics(r, lut, tile_m)
        out["metrics_in"] = m0; out["metrics_out"] = m1
    return out


# ---------------------------------------------------------------------------
# export: everything in local metres (x east, y north, z up, origin at the tile centre)
# ---------------------------------------------------------------------------

def _real_plates(tile, real: dict, cut_out):
    """the OSM street surface: the window minus the OSM blocks"""
    blocks = []
    for b in real.get("blocks", []):
        try:
            g = Polygon(b["ring"], b.get("holes") or [])
            g = g if g.is_valid else g.buffer(0)
            if not g.is_empty:
                blocks.append(g)
        except Exception:
            continue
    if not blocks:
        return []
    try:
        street = tile.difference(unary_union(blocks))
    except Exception:
        return []
    return [cut_out(g) for g in _polys(street) if g.area > 1.0]


def parts_m(res: dict, real: dict | None = None, rect=None, only: str | None = None) -> dict:
    """
    The board as shapely geometry in metres. `real` is the page's OpenStreetMap geometry
    (buildings, blocks, greens, centrelines in local metres) when an address is loaded, and
    `rect` is the part the model rebuilt, either [x0, y0, x1, y1] in metres (x east, y north)
    or any shapely polygon: the fit is used inside it, the real outlines outside, exactly what
    the viewer draws.

    `only` splits that: "fit" keeps just the piece the model made, "real" just the context
    around it. A design space writes the context once and one option model per option.
    """
    W, H, px = res["W"], res["H"], res["px_m"]
    tm = lambda p: ((p[0] - W / 2) * px, (H / 2 - p[1]) * px)  # noqa: E731

    def poly(outer, holes):
        try:
            g = Polygon([tm(p) for p in outer], [[tm(p) for p in h] for h in holes])
            return g if g.is_valid else g.buffer(0)
        except Exception:
            return Polygon()

    def poly_real(part):
        try:
            g = Polygon(part["ring"], part.get("holes") or [])
            return g if g.is_valid else g.buffer(0)
        except Exception:
            return Polygon()

    half = W * px / 2.0, H * px / 2.0
    tile = box(-half[0], -half[1], half[0], half[1])
    fitted = {
        "buildings": [(poly(b["outer"], b["holes"]), b["h"]) for b in res["buildings"]],
        "streets": [(LineString([tm(p) for p in s["pts"]]), s["w"]) for s in res["streets"] if len(s["pts"]) >= 2],
        "plates": [poly(p["outer"], p["holes"]) for p in res["plates"]],
        "blocks": [poly(p["outer"], p["holes"]) for p in res["blocks"]],
        "greens": [poly(p["outer"], p["holes"]) for p in res["greens"]],
    }
    if real is None:
        out = fitted if only != "real" else {k: [] for k in fitted}
    else:
        r = (rect if hasattr(rect, "geom_type") else box(*rect)) if rect is not None else None
        cut_in = (lambda g: g.intersection(r)) if r is not None else (lambda g: Polygon())
        cut_out = (lambda g: g.difference(r)) if r is not None else (lambda g: g)
        keep_fit = only != "real"
        keep_real = only != "fit"
        pick = lambda a, b: (a if keep_fit else []) + (b if keep_real else [])  # noqa: E731
        out = {
            "buildings": pick([(cut_in(g), h) for g, h in fitted["buildings"]],
                              [(cut_out(poly_real(b)), float(b.get("h", 7.0))) for b in real.get("buildings", [])]),
            "streets": pick([(cut_in(g), w) for g, w in fitted["streets"]],
                            [(cut_out(LineString(c["pts"])), float(c.get("w", 8.0))) for c in real.get("centrelines", []) if len(c.get("pts", [])) >= 2]),
            # the real street surface is the complement of the real blocks inside the window,
            # so the context model carries pavement, not only centrelines with a width attribute
            "plates": pick([cut_in(g) for g in fitted["plates"]], _real_plates(tile, real, cut_out)),
            "blocks": pick([cut_in(g) for g in fitted["blocks"]], [cut_out(poly_real(b)) for b in real.get("blocks", [])]),
            "greens": pick([cut_in(g) for g in fitted["greens"]], [cut_out(poly_real(b)) for b in real.get("greens", [])]),
        }

    def polys(items, with_value):
        res_ = []
        for it in items:
            g, v = (it if with_value else (it, None))
            for p in _polys(g):
                if p.area > 0.5:
                    res_.append((p, v) if with_value else p)
        return res_

    def lines(items):
        res_ = []
        for g, w in items:
            gs = [g] if isinstance(g, LineString) else list(getattr(g, "geoms", []))
            for ln in gs:
                if isinstance(ln, LineString) and ln.length > 1.0:
                    res_.append((ln, w))
        return res_

    return {"tile": tile, "buildings": polys(out["buildings"], True), "streets": lines(out["streets"]),
            "plates": polys(out["plates"], False), "blocks": polys(out["blocks"], False), "greens": polys(out["greens"], False)}


def _ring_coords(ring, ccw=True, z=0.0, nd=3):
    pts = [(round(float(x), nd), round(float(y), nd)) for x, y in ring.coords[:-1]]
    a = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    if (a < 0) == ccw:
        pts = pts[::-1]
    return pts


def to_geojson(res: dict, real: dict | None = None, rect=None, only: str | None = None) -> dict:
    P = parts_m(res, real, rect, only)
    coords = lambda g: [[list(p) for p in _ring_coords(g.exterior, True, nd=2)] + [list(_ring_coords(g.exterior, True, nd=2)[0])]] + \
        [[list(p) for p in _ring_coords(i, False, nd=2)] + [list(_ring_coords(i, False, nd=2)[0])] for i in g.interiors]  # noqa: E731
    feats = []
    for g, h in P["buildings"]:
        feats.append({"type": "Feature", "properties": {"building": "yes", "height": round(h, 2),
                                                        "building:levels": max(1, int(round(h / FLOOR_HEIGHT_M)))},
                      "geometry": {"type": "Polygon", "coordinates": coords(g)}})
    for ln, w in P["streets"]:
        feats.append({"type": "Feature", "properties": {"highway": "residential", "width": w},
                      "geometry": {"type": "LineString", "coordinates": [[round(x, 2), round(y, 2)] for x, y in ln.coords]}})
    for g in P["greens"]:
        feats.append({"type": "Feature", "properties": {"leisure": "park"}, "geometry": {"type": "Polygon", "coordinates": coords(g)}})
    for g in P["blocks"]:
        feats.append({"type": "Feature", "properties": {"block": "yes"}, "geometry": {"type": "Polygon", "coordinates": coords(g)}})
    return {"type": "FeatureCollection", "crs_note": "local metres, x east, y north, origin at tile centre",
            "tile_m": res["tile_m"], "features": feats}


LAYERS_3DM = [
    # name, colour
    ("Tile", (120, 120, 120, 255)),
    ("Buildings", (245, 245, 240, 255)),
    ("Building outlines", (90, 90, 90, 255)),
    ("Streets", (200, 40, 40, 255)),
    ("Street plates", (58, 58, 60, 255)),
    ("Blocks", (150, 150, 150, 255)),
    ("Greens", (111, 174, 69, 255)),
]


def to_3dm(res: dict, real: dict | None = None, rect=None, anchor: dict | None = None, name: str = "tile",
           only: str | None = None) -> bytes:
    """
    A Rhino .3dm (needs the `rhino3dm` package): buildings as capped extrusions with courtyards
    as inner profiles, their outlines, street centrelines with `width_m`, street plates, blocks
    and greens as closed polylines, each on its own layer, in metres. `anchor` {lat, lon} sets
    the EarthAnchorPoint so Rhino knows where the tile centre is on the planet.
    """
    return to_3dm_parts(parts_m(res, real, rect, only), anchor, name)


def clip_parts(P: dict, area, inside: bool = True) -> dict:
    """
    A parts dict cut to a polygon, or to everything outside it.

    A design space needs both halves of the same scene: the piece the model made, which is what
    one option file holds, and the surroundings, which are written once.
    """
    if area is None:
        return P if inside else {**P, "buildings": [], "streets": [], "plates": [], "blocks": [], "greens": []}
    cut = (lambda g: g.intersection(area)) if inside else (lambda g: g.difference(area))  # noqa: E731

    def polys(items, valued):
        out = []
        for it in items:
            g, v = it if valued else (it, None)
            try:
                q = cut(g)
            except Exception:
                continue
            for p in _polys(q):
                if p.area > 0.5:
                    out.append((p, v) if valued else p)
        return out

    lines = []
    for ln, w in P["streets"]:
        try:
            q = cut(ln)
        except Exception:
            continue
        for part in ([q] if isinstance(q, LineString) else list(getattr(q, "geoms", []))):
            if isinstance(part, LineString) and part.length > 1.0:
                lines.append((part, w))
    # the outline on the Tile layer follows the cut, so an option file reads as the plot it fills
    try:
        frame = P["tile"].intersection(area) if inside else P["tile"]
    except Exception:
        frame = P["tile"]
    if frame.is_empty or frame.geom_type != "Polygon":
        frame = P["tile"]
    return {"tile": frame, "buildings": polys(P["buildings"], True), "streets": lines,
            "plates": polys(P["plates"], False), "blocks": polys(P["blocks"], False),
            "greens": polys(P["greens"], False)}


def to_3dm_parts(P: dict, anchor: dict | None = None, name: str = "tile") -> bytes:
    """the writer, on geometry the caller has already composed"""
    import os
    import tempfile
    import rhino3dm as r3
    f = r3.File3dm()
    f.Settings.ModelUnitSystem = r3.UnitSystem.Meters
    f.Settings.ModelAbsoluteTolerance = 0.001
    idx = {}
    for lname, col in LAYERS_3DM:
        lay = r3.Layer(); lay.Name = lname; lay.Color = col
        idx[lname] = f.Layers.Add(lay)

    def attrs(layer, name=None, **strings):
        a = r3.ObjectAttributes(); a.LayerIndex = idx[layer]
        if name:
            a.Name = name
        for k, v in strings.items():
            a.SetUserString(k, str(v))
        return a

    def curve(ring, ccw=True, z=0.0):
        pts = _ring_coords(ring, ccw, z)
        pl = r3.Polyline([r3.Point3d(x, y, z) for x, y in pts] + [r3.Point3d(pts[0][0], pts[0][1], z)])
        return pl.ToPolylineCurve()

    def add_closed(layer, g, name=None, **strings):
        f.Objects.AddCurve(curve(g.exterior, True), attrs(layer, name, **strings))
        for i in g.interiors:
            f.Objects.AddCurve(curve(i, False), attrs(layer, (name + " hole") if name else None))

    bx = P["tile"].bounds
    add_closed("Tile", P["tile"], "tile", size_m=round(bx[2] - bx[0], 1))
    n_solid = 0
    for k, (g, h) in enumerate(P["buildings"]):
        levels = max(1, int(round(h / FLOOR_HEIGHT_M)))
        ext = r3.Extrusion.Create(curve(g.exterior, True), float(h), True)
        if ext is not None:
            for i in g.interiors:
                ext.AddInnerProfile(curve(i, False))
            f.Objects.AddExtrusion(ext, attrs("Buildings", f"building {k} ({h:.1f} m)", height_m=round(h, 2), levels=levels))
            n_solid += 1
        add_closed("Building outlines", g, f"building {k} outline", height_m=round(h, 2), levels=levels)
    for k, (ln, w) in enumerate(P["streets"]):
        pl = r3.Polyline([r3.Point3d(float(x), float(y), 0.0) for x, y in ln.coords])
        f.Objects.AddPolyline(pl, attrs("Streets", f"street {k} ({w:g} m)", width_m=w))
    for k, g in enumerate(P["plates"]):
        add_closed("Street plates", g, f"street plate {k}")
    for k, g in enumerate(P["blocks"]):
        add_closed("Blocks", g, f"block {k}", area_m2=round(g.area, 1))
    for k, g in enumerate(P["greens"]):
        add_closed("Greens", g, f"green {k}", area_m2=round(g.area, 1))
    if anchor and anchor.get("lat") is not None and anchor.get("lon") is not None:
        e = f.Settings.EarthAnchorPoint
        e.EarthBasepointLatitude = float(anchor["lat"]); e.EarthBasepointLongitude = float(anchor["lon"])
        e.EarthBasepointElevation = float(anchor.get("elevation", 0.0))
        e.Name = name; e.Description = "Urban OpenGen tile centre"
        f.Settings.EarthAnchorPoint = e
    fd, path = tempfile.mkstemp(suffix=".3dm")
    os.close(fd)
    try:
        f.Write(path, 8)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import argparse, json, time
    ap = argparse.ArgumentParser(description="vectorise tiles from an npz of [N,4,S,S] bytes")
    ap.add_argument("npz"); ap.add_argument("--lut", required=True); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="vec_out")
    args = ap.parse_args()
    lut = HeightLUT.load(args.lut)
    x = np.load(args.npz)["x"][: args.n]
    Path(args.out).mkdir(exist_ok=True)
    for i, ch in enumerate(x):
        t0 = time.time()
        res = vectorize(ch, lut)
        rt = roundtrip(ch, res, lut)
        print(i, f"{(time.time() - t0) * 1000:.0f} ms", res["stats"], {k: round(v, 3) for k, v in rt.items() if k.startswith("iou")})
        Path(args.out, f"tile_{i}.geojson").write_text(json.dumps(to_geojson(res)))

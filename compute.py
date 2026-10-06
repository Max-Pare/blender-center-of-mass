# SPDX-License-Identifier: GPL-3.0-or-later
"""Center-of-mass math.

Everything is computed in world space on the evaluated mesh (modifiers, shape
keys and object transforms applied), so non-uniform and negative scale need no
special handling.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from mathutils import Vector
from mathutils.geometry import convex_hull_2d


@dataclass
class MassResult:
    object_name: str = ""
    mode: str = "VOLUME"            # model actually used (may differ from the requested one)
    com: Vector | None = None       # world space
    volume: float = 0.0             # signed, scene units^3 (negative = normals point inward)
    area: float = 0.0               # scene units^2
    tri_count: int = 0
    open_edges: int = 0             # edges used by exactly one face
    nonmanifold_edges: int = 0      # edges used by three or more faces
    inconsistent_winding: bool = False
    fallback: str = ""              # why the requested model could not be used
    bbox_diag: float = 0.0
    # Stability, with gravity along world -Z and the object resting on its lowest point.
    ground_z: float = 0.0
    footprint: list = field(default_factory=list)  # convex hull of the contact patch, [(x, y)], CCW
    support_margin: float | None = None  # distance from CoM to footprint edge; < 0 means outside
    tip_angle: float | None = None       # radians of tilt before it topples
    pivot_edge: tuple | None = None      # ((x, y), (x, y)) footprint edge it topples over
    pivot_point: tuple | None = None     # (x, y) point of that edge nearest the plumb foot
    tip_direction: tuple | None = None   # (x, y) unit vector, the way it falls
    stale: bool = False
    error: str = ""


def read_world_mesh(ob, depsgraph):
    """Return (world vertex coords Nx3 float64, triangles Mx3 int, faces-per-edge counts)."""
    ob_eval = ob.evaluated_get(depsgraph)
    me = ob_eval.to_mesh()
    try:
        me.calc_loop_triangles()
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        edge_of_loop = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("edge_index", edge_of_loop)
        edge_use = np.bincount(edge_of_loop, minlength=len(me.edges))
        mw = np.array(ob_eval.matrix_world, dtype=np.float64)
    finally:
        ob_eval.to_mesh_clear()
    co = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
    return co, tris.reshape(-1, 3), edge_use


def _has_inconsistent_winding(tris, n_verts):
    # In a consistently wound surface every directed edge occurs at most once:
    # the neighbouring face walks the shared edge the other way round.
    t = tris.astype(np.int64)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    directed = np.concatenate((a * n_verts + b, b * n_verts + c, c * n_verts + a))
    return np.unique(directed).size != directed.size


def _xy(v):
    return float(v[0]), float(v[1])


def _stability(res, pts, contact_band):
    z = pts[:, 2]
    zmin = float(z.min())
    height = float(z.max()) - zmin
    res.ground_z = zmin

    base = pts[z <= zmin + height * contact_band, :2]
    base = np.unique(base, axis=0)
    hull = base[convex_hull_2d(base.tolist())] if len(base) > 2 else base
    res.footprint = [_xy(p) for p in hull]
    if len(hull) < 3:
        return  # point or edge contact: no support area, balance is unstable

    p = hull
    e = np.roll(hull, -1, axis=0) - p
    length = np.linalg.norm(e, axis=1)
    keep = length > 1e-12
    p, e, length = p[keep], e[keep], length[keep]
    outward = np.stack((e[:, 1], -e[:, 0]), axis=1) / length[:, None]  # right side of a CCW edge
    c2 = np.array(res.com[:2])
    signed = ((c2 - p) * outward).sum(axis=1)  # > 0 outside that edge
    # The edge the plumb foot is closest to (or furthest beyond) is the weak side.
    i = int(signed.argmax())
    res.support_margin = float(-signed[i])
    res.pivot_edge = (_xy(p[i]), _xy(p[i] + e[i]))
    t = float(np.clip(np.dot(c2 - p[i], e[i]) / length[i] ** 2, 0.0, 1.0))
    res.pivot_point = _xy(p[i] + t * e[i])
    res.tip_direction = _xy(outward[i])

    com_height = res.com.z - zmin
    if com_height > 0.0:
        res.tip_angle = math.atan2(res.support_margin, com_height)


def compute(ob, depsgraph, mode, contact_band=0.005):
    """Center of mass of `ob` for a uniform-density model.

    mode: 'VOLUME'   solid body (divergence theorem over signed tetrahedra)
          'SURFACE'  thin shell of constant thickness (area-weighted triangles)
          'VERTICES' plain vertex average, for comparison
    contact_band: fraction of object height counted as touching the ground.
    """
    res = MassResult(object_name=ob.name, mode=mode)
    co, tris, edge_use = read_world_mesh(ob, depsgraph)
    if len(co) == 0:
        res.error = "Mesh has no vertices"
        return res

    res.tri_count = len(tris)
    res.open_edges = int(np.count_nonzero(edge_use == 1))
    res.nonmanifold_edges = int(np.count_nonzero(edge_use > 2))

    used = co[np.unique(tris)] if len(tris) else co
    res.bbox_diag = float(np.linalg.norm(np.ptp(used, axis=0)))

    if len(tris):
        res.inconsistent_winding = _has_inconsistent_winding(tris, len(co))
        # Work relative to the centroid: keeps the tetrahedra small and the sums well conditioned.
        origin = used.mean(axis=0)
        a = co[tris[:, 0]] - origin
        b = co[tris[:, 1]] - origin
        c = co[tris[:, 2]] - origin
        abc = a + b + c

        six_vol = np.einsum("ij,ij->i", a, np.cross(b, c))
        res.volume = float(six_vol.sum()) / 6.0
        twice_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        res.area = float(twice_area.sum()) / 2.0

        if mode == "VOLUME" and abs(res.volume) <= 1e-9 * max(res.bbox_diag, 1e-30) ** 3:
            res.mode = mode = "SURFACE"
            res.fallback = "Mesh encloses no volume, using the surface"
        if mode == "SURFACE" and res.area <= 0.0:
            res.mode = mode = "VERTICES"
            res.fallback = "Mesh has no surface area, using the vertices"

        if mode == "VOLUME":
            res.com = Vector(origin + (six_vol @ abc) / (4.0 * six_vol.sum()))
        elif mode == "SURFACE":
            res.com = Vector(origin + (twice_area @ abc) / (3.0 * twice_area.sum()))
    elif mode != "VERTICES":
        res.mode = "VERTICES"
        res.fallback = "Mesh has no faces, using the vertices"

    if res.com is None:
        res.com = Vector(co.mean(axis=0))

    _stability(res, used, contact_band)
    return res

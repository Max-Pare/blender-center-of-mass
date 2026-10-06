# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport overlay: CoM symbol, axes, plumb line and support footprint."""

import math

import blf
import bpy
import gpu
from bpy_extras.view3d_utils import location_3d_to_region_2d
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

AXIS_COLORS = ((1.0, 0.21, 0.33, 1.0), (0.54, 0.86, 0.0, 1.0), (0.17, 0.56, 1.0, 1.0))
STABLE_COLOR = (0.35, 0.9, 0.45, 0.9)
UNSTABLE_COLOR = (1.0, 0.3, 0.25, 0.9)
NEUTRAL_COLOR = (1.0, 0.8, 0.2, 0.9)
PIVOT_COLOR = (1.0, 0.3, 0.85, 1.0)  # magenta: clear of selection orange, axes and status colors
LEVER_COLOR = (1.0, 0.3, 0.85, 0.6)
DARK = (0.05, 0.05, 0.05, 1.0)
LIGHT = (0.95, 0.95, 0.95, 1.0)

_shaders = {}


def _shader(name):
    # Built-in shaders need a GPU context, so create them on first draw, never at register time.
    shader = _shaders.get(name)
    if shader is None:
        shader = _shaders[name] = gpu.shader.from_builtin(name)
    return shader


def _lines(region, coords, color, width):
    shader = _shader("POLYLINE_UNIFORM_COLOR")
    batch = batch_for_shader(shader, "LINES", {"pos": coords})
    shader.bind()
    shader.uniform_float("viewportSize", (region.width, region.height))
    shader.uniform_float("lineWidth", width)
    shader.uniform_float("color", color)
    batch.draw(shader)


def _tris(coords, indices, color):
    shader = _shader("UNIFORM_COLOR")
    batch = batch_for_shader(shader, "TRIS", {"pos": coords}, indices=indices)
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _world_per_pixel(region, rv3d, loc):
    # A step along the view's right axis keeps the depth, so the projection is linear.
    right = rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))
    p0 = location_3d_to_region_2d(region, rv3d, loc)
    p1 = location_3d_to_region_2d(region, rv3d, loc + right)
    if p0 is None or p1 is None:
        return None
    pixels = (p1 - p0).length
    return 1.0 / pixels if pixels > 1e-9 else None


def _status_color(res):
    if res.support_margin is None:
        return NEUTRAL_COLOR if res.footprint else UNSTABLE_COLOR
    return STABLE_COLOR if res.support_margin > 0.0 else UNSTABLE_COLOR


class _GPUState:
    def __enter__(self):
        self.blend = gpu.state.blend_get()
        self.depth = gpu.state.depth_test_get()
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")  # the CoM is inside the mesh: always draw in front

    def __exit__(self, *exc):
        gpu.state.blend_set(self.blend)
        gpu.state.depth_test_set(self.depth)


def draw_view(get_state):
    state = get_state(bpy.context)
    if state is None:
        return
    res, settings, _label = state
    region, rv3d = bpy.context.region, bpy.context.region_data
    com = res.com
    wpp = _world_per_pixel(region, rv3d, com)
    if wpp is None:
        return
    ui_scale = bpy.context.preferences.system.ui_scale

    with _GPUState():
        if settings.show_axes:
            half = settings.marker_size * 2.6 * ui_scale * wpp
            for axis, color in enumerate(AXIS_COLORS):
                offset = Vector((0.0, 0.0, 0.0))
                offset[axis] = half
                _lines(region, [com - offset, com + offset], color, 2.0 * ui_scale)

        if settings.show_stability and res.footprint:
            color = _status_color(res)
            z = res.ground_z
            foot = Vector((com.x, com.y, z))
            _lines(region, [com, foot], color, 1.5 * ui_scale)

            ring = [Vector((x, y, z)) for x, y in res.footprint]
            if len(ring) > 1:
                edges = []
                for i, p in enumerate(ring):
                    edges += (p, ring[(i + 1) % len(ring)])
                _lines(region, edges, color, 2.0 * ui_scale)

            tick = 5.0 * ui_scale * (_world_per_pixel(region, rv3d, foot) or wpp)
            dx, dy = Vector((tick, 0.0, 0.0)), Vector((0.0, tick, 0.0))
            _lines(region, [foot - dx, foot + dx, foot - dy, foot + dy], color, 2.0 * ui_scale)

            if res.pivot_edge is not None:
                _draw_tip(region, rv3d, res, com, foot, ui_scale)


def _draw_tip(region, rv3d, res, com, foot, ui_scale):
    """Weak side: the edge it topples over, the lever triangle that sets the angle, a fall arrow."""
    z = res.ground_z
    a, b = (Vector((x, y, z)) for x, y in res.pivot_edge)
    pivot = Vector((*res.pivot_point, z))
    out = Vector((*res.tip_direction, 0.0))
    side = Vector((-out.y, out.x, 0.0))

    _lines(region, [a, b], PIVOT_COLOR, 4.0 * ui_scale)
    # foot -> pivot is the margin d, pivot -> CoM leans at the tip angle from vertical.
    _lines(region, [foot, pivot, pivot, com], LEVER_COLOR, 1.5 * ui_scale)

    wpp = _world_per_pixel(region, rv3d, pivot)
    if wpp is None:
        return
    length = 60.0 * ui_scale * wpp
    head = 15.0 * ui_scale * wpp
    tip = pivot + out * length
    neck = tip - out * head
    _lines(region, [pivot, neck], PIVOT_COLOR, 2.5 * ui_scale)
    _tris([tip, neck + side * head * 0.55, neck - side * head * 0.55], [(0, 1, 2)], PIVOT_COLOR)


def _com_symbol(cx, cy, radius, ui_scale, segments=10):
    """The usual center-of-mass glyph: a disc with alternating dark and light quadrants."""
    for quadrants, color in (((0, 2), DARK), ((1, 3), LIGHT)):
        coords, indices = [], []
        for q in quadrants:
            base = len(coords)
            coords.append((cx, cy, 0.0))
            for i in range(segments + 1):
                a = (q + i / segments) * math.pi / 2.0
                coords.append((cx + radius * math.cos(a), cy + radius * math.sin(a), 0.0))
            indices += [(base, base + i, base + i + 1) for i in range(1, segments + 1)]
        _tris(coords, indices, color)

    region = bpy.context.region
    ring = []
    steps = segments * 4
    for i in range(steps):
        for j in (i, i + 1):
            a = 2.0 * math.pi * j / steps
            ring.append((cx + radius * math.cos(a), cy + radius * math.sin(a), 0.0))
    _lines(region, ring, DARK, 1.5 * ui_scale)


def draw_pixel(get_state):
    state = get_state(bpy.context)
    if state is None:
        return
    res, settings, label = state
    region, rv3d = bpy.context.region, bpy.context.region_data
    p = location_3d_to_region_2d(region, rv3d, res.com)
    if p is None:
        return
    ui_scale = bpy.context.preferences.system.ui_scale
    radius = settings.marker_size * ui_scale

    with _GPUState():
        _com_symbol(p.x, p.y, radius, ui_scale)

    if settings.show_label and label:
        font = 0
        blf.size(font, 11.0 * ui_scale)
        blf.color(font, 1.0, 1.0, 1.0, 1.0)
        blf.enable(font, blf.SHADOW)
        blf.shadow(font, 6, 0.0, 0.0, 0.0, 0.85)
        x = p.x + (radius * 2.6 if settings.show_axes else radius) + 4.0 * ui_scale
        y = p.y + radius * 0.4
        for line in label:
            blf.position(font, x, y, 0.0)
            blf.draw(font, line)
            y -= 14.0 * ui_scale
        blf.disable(font, blf.SHADOW)

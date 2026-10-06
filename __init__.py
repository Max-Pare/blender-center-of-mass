# SPDX-License-Identifier: GPL-3.0-or-later
"""Center of Mass Visualizer.

Draws the physical center of mass of a mesh object in the 3D viewport and
reports volume, mass and tip-over stability. Lives in View3D > Sidebar > Mass.
"""

import math

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

from . import compute, draw

# Densities in kg/m^3.
MATERIALS = (
    ("PLA", "PLA", 1240.0),
    ("PETG", "PETG", 1270.0),
    ("ABS", "ABS", 1040.0),
    ("ASA", "ASA", 1070.0),
    ("TPU", "TPU", 1210.0),
    ("NYLON", "Nylon PA12", 1010.0),
    ("RESIN", "Standard Resin", 1180.0),
    ("WATER", "Water", 1000.0),
    ("WOOD", "Pine Wood", 500.0),
    ("ALUMINIUM", "Aluminium", 2700.0),
    ("STEEL", "Steel", 7850.0),
)
DENSITY = {key: density for key, _name, density in MATERIALS}

# Results keyed by (object name, model, contact band). Several 3D views or
# windows can show different objects, so this is a small dict, not one slot.
_results = {}
_update_pending = False
_draw_handles = []


# ------------------------------------------------------------------ helpers

def _target(settings, view_layer):
    ob = settings.target or view_layer.objects.active
    return ob if ob is not None and ob.type == "MESH" else None


def _key(ob, settings):
    return (ob.name_full, settings.mode, round(settings.contact_band, 6))


def _compute(ob, depsgraph, settings):
    try:
        return compute.compute(ob, depsgraph, settings.mode, settings.contact_band / 100.0)
    except Exception as ex:  # a bad mesh must never break viewport drawing
        return compute.MassResult(object_name=ob.name, error=f"{type(ex).__name__}: {ex}")


def _store(key, res):
    if len(_results) > 32:
        _results.clear()
    _results[key] = res


def _tag_redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _run_update():
    global _update_pending
    _update_pending = False
    wm = bpy.context.window_manager
    if wm is None:
        return None
    for win in wm.windows:
        settings = win.scene.com_viz
        if not settings.enabled:
            continue
        ob = _target(settings, win.view_layer)
        if ob is None:
            continue
        key = _key(ob, settings)
        res = _results.get(key)
        if res is not None and not (res.stale and settings.live_update):
            continue
        with bpy.context.temp_override(window=win):
            _store(key, _compute(ob, bpy.context.evaluated_depsgraph_get(), settings))
    _tag_redraw()
    return None


def _request_update(delay=0.0):
    """Recompute from a timer: drawing code must never evaluate the depsgraph."""
    global _update_pending
    if not _update_pending:
        _update_pending = True
        bpy.app.timers.register(_run_update, first_interval=delay)


def _density(settings):
    return settings.density if settings.material == "CUSTOM" else DENSITY[settings.material]


def _unit_scale(scene):
    units = scene.unit_settings
    return units.scale_length if units.system != "NONE" else 1.0


def _mass_kg(res, settings, scene):
    s = _unit_scale(scene)
    if res.mode == "VOLUME":
        return abs(res.volume) * s ** 3 * _density(settings)
    if res.mode == "SURFACE":
        return res.area * s ** 2 * settings.shell_thickness * s * _density(settings)
    return None


def _fmt(scene, category, value, precision=3):
    """Format a value given in SI base units (m, m², m³, kg) the way the scene displays units."""
    system = scene.unit_settings.system
    if system == "NONE":
        if category == "MASS":
            system = "METRIC"
        else:
            return f"{value:.{precision + 2}g}"
    return bpy.utils.units.to_string(system, category, value, precision=precision)


def _fmt_length(scene, value):
    return _fmt(scene, "LENGTH", value * _unit_scale(scene))


def _label(res, settings, scene):
    mass = _mass_kg(res, settings, scene)
    lines = ["CoM  " + _fmt(scene, "MASS", mass) if mass is not None else "CoM"]
    if settings.show_stability and res.footprint:
        if res.support_margin is not None and res.support_margin > 0.0 and res.tip_angle is not None:
            lines.append(f"tips at {math.degrees(res.tip_angle):.1f}°")
        else:
            lines.append("unstable")
    if res.stale:
        lines.append("(outdated)")
    return lines


def _draw_state(context):
    """What the overlay should draw in this 3D view, or None."""
    space = context.space_data
    if space is None or not space.overlay.show_overlays:
        return None
    settings = context.scene.com_viz
    if not settings.enabled:
        return None
    ob = _target(settings, context.view_layer)
    if ob is None or not ob.visible_get(viewport=space):
        return None
    res = _results.get(_key(ob, settings))
    if res is None:
        _request_update()
        return None
    if res.com is None:
        return None
    return res, settings, _label(res, settings, context.scene)


def _context_result(context, compute_missing=True):
    settings = context.scene.com_viz
    ob = _target(settings, context.view_layer)
    if ob is None:
        return None, None
    key = _key(ob, settings)
    res = _results.get(key)
    if compute_missing and (res is None or res.stale):
        res = _compute(ob, context.evaluated_depsgraph_get(), settings)
        _store(key, res)
    return ob, res


# ------------------------------------------------------------------ handlers

@persistent
def _on_depsgraph_update(scene, depsgraph):
    if not _results:
        return
    changed = {
        u.id.original.name_full
        for u in depsgraph.updates
        if isinstance(u.id, bpy.types.Object) and (u.is_updated_geometry or u.is_updated_transform)
    }
    if not changed:
        return
    hit = False
    for key, res in _results.items():
        if key[0] in changed and not res.stale:
            res.stale = True
            hit = True
    if hit:
        if scene.com_viz.live_update:
            _request_update()
        else:
            _tag_redraw()


@persistent
def _on_load_post(*_args):
    _results.clear()


# ------------------------------------------------------------------ settings

def _settings_changed(_self, _context):
    _request_update()
    _tag_redraw()


def _display_changed(_self, _context):
    _tag_redraw()


class COMVIZ_Settings(PropertyGroup):
    enabled: BoolProperty(
        name="Show Center of Mass",
        description="Compute the center of mass and draw it in the 3D viewport",
        default=False,
        update=_settings_changed,
    )
    target: PointerProperty(
        name="Object",
        description="Mesh to analyse. Leave empty to follow the active object",
        type=bpy.types.Object,
        poll=lambda _self, ob: ob.type == "MESH",
        update=_settings_changed,
    )
    mode: EnumProperty(
        name="Model",
        description="How mass is distributed over the mesh",
        items=(
            ("VOLUME", "Solid",
             "Uniform-density solid filling the mesh. Needs a closed mesh. "
             "A print with sparse infill is lighter inside than this model"),
            ("SURFACE", "Shell",
             "Thin shell of constant thickness over the surface. For open meshes, sheet parts, hollow casts"),
            ("VERTICES", "Vertices",
             "Plain average of the vertex positions. Not physical: depends on mesh density, for comparison only"),
        ),
        default="VOLUME",
        update=_settings_changed,
    )
    material: EnumProperty(
        name="Material",
        description="Density used for the mass",
        items=[(key, name, f"{density:g} kg/m³") for key, name, density in MATERIALS]
        + [("CUSTOM", "Custom", "Enter the density by hand")],
        default="PLA",
        update=_display_changed,
    )
    density: FloatProperty(
        name="Density",
        description="Custom density in kg/m³",
        default=1000.0,
        min=0.0,
        soft_max=20000.0,
        precision=1,
        update=_display_changed,
    )
    shell_thickness: FloatProperty(
        name="Thickness",
        description="Wall thickness of the shell, for the mass only (the center does not depend on it)",
        default=0.002,
        min=0.0,
        subtype="DISTANCE",
        unit="LENGTH",
        update=_display_changed,
    )
    contact_band: FloatProperty(
        name="Contact Band",
        description="Vertices this close to the lowest point, as a percentage of the object's height, "
                    "count as touching the ground",
        default=0.5,
        min=0.0,
        max=20.0,
        precision=2,
        subtype="PERCENTAGE",
        update=_settings_changed,
    )
    live_update: BoolProperty(
        name="Live Update",
        description="Recompute whenever the object changes. Turn off for very heavy meshes",
        default=True,
        update=_settings_changed,
    )
    show_axes: BoolProperty(name="Axes", description="Draw X/Y/Z axes through the center",
                            default=True, update=_display_changed)
    show_label: BoolProperty(name="Label", description="Write mass and stability next to the marker",
                             default=True, update=_display_changed)
    show_stability: BoolProperty(
        name="Stability",
        description="Draw the plumb line and the footprint the object stands on, "
                    "green when it stands, red when it topples (gravity along -Z)",
        default=True,
        update=_display_changed,
    )
    marker_size: IntProperty(name="Marker Size", description="Radius of the marker in pixels",
                             default=9, min=4, max=40, subtype="PIXEL", update=_display_changed)


# ------------------------------------------------------------------ operators

class COMVIZ_OT_recompute(Operator):
    bl_idname = "view3d.com_recompute"
    bl_label = "Recompute"
    bl_description = "Recompute the center of mass now"

    @classmethod
    def poll(cls, context):
        return _target(context.scene.com_viz, context.view_layer) is not None

    def execute(self, context):
        settings = context.scene.com_viz
        ob = _target(settings, context.view_layer)
        _store(_key(ob, settings), _compute(ob, context.evaluated_depsgraph_get(), settings))
        _tag_redraw()
        return {"FINISHED"}


class _ResultOperator:
    @classmethod
    def poll(cls, context):
        return _target(context.scene.com_viz, context.view_layer) is not None

    def result(self, context):
        ob, res = _context_result(context)
        if res is None or res.com is None:
            self.report({"ERROR"}, res.error if res is not None and res.error else "No center of mass")
            return ob, None
        return ob, res


class COMVIZ_OT_cursor_to_com(_ResultOperator, Operator):
    bl_idname = "view3d.com_cursor_to_com"
    bl_label = "Cursor to CoM"
    bl_description = "Snap the 3D cursor to the center of mass"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        _ob, res = self.result(context)
        if res is None:
            return {"CANCELLED"}
        context.scene.cursor.location = res.com
        return {"FINISHED"}


class COMVIZ_OT_add_empty(_ResultOperator, Operator):
    bl_idname = "object.com_add_empty"
    bl_label = "Add Empty at CoM"
    bl_description = "Add an empty at the current center of mass (a static marker, it does not follow edits)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ob, res = self.result(context)
        if res is None:
            return {"CANCELLED"}
        empty = bpy.data.objects.new(f"{ob.name}_CoM", None)
        empty.empty_display_type = "SPHERE"
        empty.empty_display_size = max(res.bbox_diag * 0.03, 1e-4)
        empty.location = res.com
        context.collection.objects.link(empty)
        self.report({"INFO"}, f"Added {empty.name}")
        return {"FINISHED"}


class COMVIZ_OT_origin_to_com(_ResultOperator, Operator):
    bl_idname = "object.com_origin_to_com"
    bl_label = "Origin to CoM"
    bl_description = (
        "Move the object's origin to the center of mass, with modifiers applied. "
        "Modifiers that depend on the origin (Mirror, Array, Screw) may change the shape"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and super().poll(context)

    def execute(self, context):
        ob, res = self.result(context)
        if res is None:
            return {"CANCELLED"}
        cursor = context.scene.cursor
        saved = cursor.location.copy()
        cursor.location = res.com
        try:
            with context.temp_override(object=ob, active_object=ob,
                                       selected_objects=[ob], selected_editable_objects=[ob]):
                bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        finally:
            cursor.location = saved
        return {"FINISHED"}


# ------------------------------------------------------------------ panels

def _value_row(layout, label, value, icon="NONE"):
    split = layout.split(factor=0.4)
    left = split.row()
    left.alignment = "RIGHT"
    left.label(text=label)
    split.label(text=value, icon=icon)


def _panel_result(context):
    settings = context.scene.com_viz
    if not settings.enabled:
        return None
    ob = _target(settings, context.view_layer)
    return None if ob is None else _results.get(_key(ob, settings))


class COMVIZ_PT_main(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mass"
    bl_label = "Center of Mass"

    def draw_header(self, context):
        self.layout.prop(context.scene.com_viz, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        settings = scene.com_viz

        col = layout.column()
        col.prop(settings, "target")
        if settings.target is None:
            active = context.view_layer.objects.active
            hint = f"Following {active.name}" if active is not None else "No active object"
            if active is not None and active.type != "MESH":
                hint = f"{active.name} is not a mesh"
            _value_row(col, "", hint, icon="RESTRICT_SELECT_OFF")
        col.prop(settings, "mode")
        if settings.mode != "VERTICES":
            col.prop(settings, "material")
            if settings.material == "CUSTOM":
                col.prop(settings, "density")
            else:
                _value_row(col, "Density", f"{_density(settings):g} kg/m³")
            if settings.mode == "SURFACE":
                col.prop(settings, "shell_thickness")

        if not settings.enabled:
            layout.prop(settings, "enabled", toggle=True, icon="HIDE_OFF")
            return

        ob = _target(settings, context.view_layer)
        if ob is None:
            layout.label(text="Select a mesh object", icon="INFO")
            return
        res = _results.get(_key(ob, settings))
        if res is None:
            _request_update()
            layout.label(text="Computing...", icon="TIME")
            return
        if res.error:
            layout.label(text=res.error, icon="ERROR")
            return

        box = layout.box()
        col = box.column(align=True)
        for axis, value in zip("XYZ", res.com):
            _value_row(col, axis, _fmt_length(scene, value))
        col.separator()
        s = _unit_scale(scene)
        if res.mode == "VOLUME":
            _value_row(col, "Volume", _fmt(scene, "VOLUME", abs(res.volume) * s ** 3))
        else:
            _value_row(col, "Area", _fmt(scene, "AREA", res.area * s ** 2))
        mass = _mass_kg(res, settings, scene)
        if mass is not None:
            _value_row(col, "Mass", _fmt(scene, "MASS", mass))
        _value_row(col, "Triangles", f"{res.tri_count:,}")

        warnings = box.column(align=True)
        if res.fallback:
            warnings.label(text=res.fallback, icon="ERROR")
        if res.mode == "VOLUME":
            if res.open_edges:
                warnings.label(text=f"{res.open_edges:,} open edges: approximate", icon="ERROR")
            if res.nonmanifold_edges:
                warnings.label(text=f"{res.nonmanifold_edges:,} non-manifold edges", icon="ERROR")
            elif res.inconsistent_winding:
                warnings.label(text="Inconsistent normals: result is wrong", icon="ERROR")
                warnings.label(text="Fix with Mesh > Normals > Recalculate Outside")
        if res.stale:
            warnings.label(text="Outdated", icon="TIME")

        row = layout.row(align=True)
        row.operator(COMVIZ_OT_cursor_to_com.bl_idname, text="Cursor", icon="PIVOT_CURSOR")
        row.operator(COMVIZ_OT_add_empty.bl_idname, text="Empty", icon="EMPTY_AXIS")
        row.operator(COMVIZ_OT_origin_to_com.bl_idname, text="Origin", icon="OBJECT_ORIGIN")
        row.operator(COMVIZ_OT_recompute.bl_idname, text="", icon="FILE_REFRESH")


class COMVIZ_PT_stability(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mass"
    bl_label = "Stability"
    bl_parent_id = "COMVIZ_PT_main"

    def draw_header(self, context):
        self.layout.prop(context.scene.com_viz, "show_stability", text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        scene = context.scene
        settings = scene.com_viz
        layout.active = settings.show_stability

        res = _panel_result(context)
        if res is not None and res.com is not None and not res.error:
            col = layout.box().column(align=True)
            if res.support_margin is None:
                col.label(text="Rests on a point or an edge: unstable", icon="ERROR")
            elif res.support_margin > 0.0:
                col.label(text="Stands", icon="CHECKMARK")
                if res.tip_angle is not None:
                    _value_row(col, "Tips at", f"{math.degrees(res.tip_angle):.1f}° tilt")
                _value_row(col, "Margin", _fmt_length(scene, res.support_margin))
            else:
                col.label(text="Topples: CoM outside footprint", icon="ERROR")
                _value_row(col, "Overhang", _fmt_length(scene, -res.support_margin))
            _value_row(col, "CoM Height", _fmt_length(scene, res.com.z - res.ground_z))

        layout.prop(settings, "contact_band")
        layout.label(text="Gravity -Z, resting on the lowest point", icon="INFO")


class COMVIZ_PT_display(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mass"
    bl_label = "Display"
    bl_parent_id = "COMVIZ_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        settings = context.scene.com_viz
        col = layout.column(heading="Show")
        col.prop(settings, "show_axes")
        col.prop(settings, "show_label")
        layout.prop(settings, "marker_size")
        layout.prop(settings, "live_update")


# ------------------------------------------------------------------ registration

_classes = (
    COMVIZ_Settings,
    COMVIZ_OT_recompute,
    COMVIZ_OT_cursor_to_com,
    COMVIZ_OT_add_empty,
    COMVIZ_OT_origin_to_com,
    COMVIZ_PT_main,
    COMVIZ_PT_stability,
    COMVIZ_PT_display,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.com_viz = PointerProperty(type=COMVIZ_Settings)
    space = bpy.types.SpaceView3D
    _draw_handles.append(space.draw_handler_add(draw.draw_view, (_draw_state,), "WINDOW", "POST_VIEW"))
    _draw_handles.append(space.draw_handler_add(draw.draw_pixel, (_draw_state,), "WINDOW", "POST_PIXEL"))
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.load_post.append(_on_load_post)


def unregister():
    global _update_pending
    for handle in _draw_handles:
        bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
    _draw_handles.clear()
    if _on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_on_depsgraph_update)
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    if bpy.app.timers.is_registered(_run_update):
        bpy.app.timers.unregister(_run_update)
    _update_pending = False
    _results.clear()
    del bpy.types.Scene.com_viz
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)

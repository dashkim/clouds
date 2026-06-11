"""VTK scene for patch terrain mesh + pseudo-3D scalar slab."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import vtk

from lib.terrain_mesh import (
    PatchGrid,
    build_feature_terrain_surface,
    build_inversion_terrain_overlay,
    build_scalar_slab,
    build_terrain_surface,
    lonlat_to_local_meters,
    patch_centroid_metric,
    patch_horizontal_span_m,
)
from lib.vtk_alps_scene import _cloud_lut, _terrain_lut

import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

BG_COLOR = (0.027, 0.043, 0.078)

SLAB_SCALAR_NAMES = {
    "clt": "clt",
    "pressure": "pressure_height",
    "wind": "wind_speed",
    "inversion": "inversion",
    "inversion_compare": "agreement",
}


def _pressure_lut(vmin: float, vmax: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(vmin, vmax)
    for i in range(256):
        t = i / 255.0
        r = 0.35 + 0.65 * t
        g = 0.55 + 0.4 * t
        b = 0.95 - 0.35 * t
        alpha = max(0.0, min(1.0, (t - 0.02) / 0.98)) * 0.88
        lut.SetTableValue(i, r, g, b, alpha)
    lut.Build()
    return lut


def _wind_lut(vmin: float, vmax: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(vmin, vmax)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            u = t / 0.5
            r, g, b = 0.2 + 0.6 * u, 0.65 + 0.25 * u, 0.25
        else:
            u = (t - 0.5) / 0.5
            r, g, b = 0.8 + 0.2 * u, 0.9 - 0.5 * u, 0.25 - 0.15 * u
        alpha = max(0.35, min(1.0, 0.35 + 0.65 * t))
        lut.SetTableValue(i, r, g, b, alpha)
    lut.Build()
    return lut


def _inversion_fog_lut(vmin: float, vmax: float, *, max_alpha: float = 0.92) -> vtk.vtkLookupTable:
    """Opaque fog bank for horizontal inversion cloud layer."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(vmin, vmax)
    for i in range(256):
        t = i / 255.0
        if t < 0.08:
            alpha = 0.0
        else:
            alpha = min(max_alpha, 0.12 + max_alpha * 0.85 * t)
        r = 0.82 + 0.12 * t
        g = 0.86 + 0.1 * t
        b = 0.95
        lut.SetTableValue(i, r, g, b, alpha)
    lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
    lut.Build()
    return lut


def _inversion_ridge_lut() -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0.0, 1.0)
    for i in range(256):
        t = i / 255.0
        if t < 0.45:
            alpha = 0.0
            r, g, b = 0.9, 0.75, 0.2
        else:
            alpha = min(1.0, (t - 0.45) / 0.55)
            r = 0.95 + 0.05 * t
            g = 0.72 + 0.2 * t
            b = 0.15 + 0.1 * t
        lut.SetTableValue(i, r, g, b, alpha)
    lut.Build()
    return lut


def _deck_gap_lut() -> vtk.vtkLookupTable:
    """Gap between decks: 1=TP green, 2=FP red, 3=FN blue."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0.0, 3.0)
    colors = {
        0: (0.0, 0.0, 0.0, 0.0),
        1: (0.15, 0.88, 0.38, 0.72),
        2: (0.96, 0.22, 0.18, 0.82),
        3: (0.22, 0.48, 0.98, 0.82),
    }
    for i in range(256):
        bucket = max(0, min(3, int(round(i / 255.0 * 3.0))))
        lut.SetTableValue(i, *colors[bucket])
    lut.Build()
    return lut


def _opaque_feature_lut(
    vmin: float,
    vmax: float,
    *,
    mode: str,
) -> vtk.vtkLookupTable:
    """Opaque terrain shading LUT: hue encodes value."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(vmin, vmax)
    for i in range(256):
        t = i / 255.0
        if mode == "lcc":
            if t < 0.05:
                r, g, b, a = 0.22, 0.28, 0.18, 1.0
            else:
                r = 0.22 + 0.35 * t
                g = 0.28 + 0.42 * t
                b = 0.18 + 0.55 * t
                a = 1.0
        elif mode == "t2m":
            r = 0.12 + 0.88 * t
            g = 0.35 + 0.45 * (1.0 - abs(t - 0.5) * 2.0)
            b = 0.95 - 0.8 * t
            a = 1.0
        else:  # wind
            if t < 0.5:
                u = t / 0.5
                r, g, b = 0.2 + 0.6 * u, 0.65 + 0.25 * u, 0.25
            else:
                u = (t - 0.5) / 0.5
                r, g, b = 0.8 + 0.2 * u, 0.9 - 0.5 * u, 0.25 - 0.15 * u
            a = 1.0
        lut.SetTableValue(i, r, g, b, a)
    lut.Build()
    return lut


def _deck_surface_lut(r: float, g: float, b: float, *, alpha: float) -> vtk.vtkLookupTable:
    """Binary deck visibility: 0=hidden, 1=tinted surface."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(0.0, 1.0)
    lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
    lut.SetTableValue(255, r, g, b, alpha)
    for i in range(1, 255):
        t = i / 255.0
        lut.SetTableValue(i, r, g, b, alpha * t)
    lut.Build()
    return lut


def _slab_lut(slab_mode: str, vmin: float, vmax: float) -> vtk.vtkLookupTable:
    if slab_mode == "pressure":
        return _pressure_lut(vmin, vmax)
    if slab_mode == "wind":
        return _wind_lut(vmin, vmax)
    if slab_mode == "inversion":
        return _inversion_ridge_lut()
    if slab_mode == "inversion_compare":
        return _deck_gap_lut()
    return _cloud_lut(vmin, vmax)


def _slab_bar_title(slab_mode: str) -> str:
    if slab_mode == "pressure":
        return "Cloud base height (m)"
    if slab_mode == "wind":
        return "Wind speed (m/s)"
    if slab_mode == "inversion":
        return "Inversion mask"
    if slab_mode == "inversion_compare":
        return "Agreement"
    return "clt (%)"


@dataclass
class PatchSceneConfig:
    vert_exag: float = 5.0
    slab_height: float = 400.0
    slab_mode: str = "clt"
    orbit_deg: float = 45.0
    orbit_start_deg: float = -50.0
    camera_view: str = "oblique"
    elevation_angle_deg: float = 36.0
    camera_elev_percentile: float = 24.0
    focal_elev_percentile: float = 71.0
    orbit_radius_factor: float = 1.15
    slab_vmin: float | None = None
    slab_vmax: float | None = None
    wind_glyph_stride: int = 4
    show_scalar_bars: bool = True
    cloud_deck_margin_m: float = 50.0
    show_cloud_deck: bool = True
    show_ridge_highlight: bool = False
    cloud_deck_max_alpha: float = 0.92
    compare_deck_opacity: float = 0.62
    compare_gap_opacity: float = 0.78
    terrain_opacity_compare: float = 0.88
    feature_lcc_height: float = 180.0
    feature_t2m_height: float = 360.0
    feature_wind_height: float = 360.0
    feature_slab_opacity: float = 0.88
    feature_overlay_kind: str = "lcc"

    @property
    def clt_vmin(self) -> float:
        return self.slab_vmin if self.slab_vmin is not None else 5.0

    @clt_vmin.setter
    def clt_vmin(self, value: float) -> None:
        self.slab_vmin = value

    @property
    def clt_vmax(self) -> float:
        return self.slab_vmax if self.slab_vmax is not None else 95.0

    @clt_vmax.setter
    def clt_vmax(self, value: float) -> None:
        self.slab_vmax = value


class Patch3DScene:
    """Metric-space terrain + scalar slab with orbit camera."""

    def __init__(
        self,
        grid: PatchGrid,
        config: PatchSceneConfig | None = None,
        *,
        region_title: str = "Alps patch",
    ) -> None:
        if grid.overlay is None:
            raise ValueError("PatchGrid must include overlay series for Patch3DScene.")

        self.grid = grid
        self.overlay_series = grid.overlay
        self.config = config or PatchSceneConfig(slab_mode=grid.slab_mode)
        self.config.slab_mode = grid.slab_mode
        self.region_title = region_title
        self._scalar_name = SLAB_SCALAR_NAMES.get(grid.slab_mode, "scalar")

        z_valid = grid.z[np.isfinite(grid.z)]
        if z_valid.size == 0:
            raise ValueError("No valid terrain in patch.")

        self.elev_vmin = float(np.nanpercentile(z_valid, 8))
        self.elev_vmax = float(np.nanpercentile(z_valid, 99.5))
        self._init_slab_range()

        self._centroid = patch_centroid_metric(grid, self.config.vert_exag)
        self._orbit_radius = self.config.orbit_radius_factor * patch_horizontal_span_m(grid)
        ve = self.config.vert_exag
        self._camera_z = float(
            np.nanpercentile(z_valid, self.config.camera_elev_percentile) * ve,
        )
        self._focal_z = float(
            np.nanpercentile(z_valid, self.config.focal_elev_percentile) * ve,
        )

        self._is_feature_overlay_scene = grid.slab_mode == "feature_overlay"
        self._is_inversion_scene = grid.slab_mode in (
            "inversion", "inversion_compare", "feature_overlay",
        )
        self._is_compare_scene = grid.slab_mode == "inversion_compare"

        self._feature_kind = self.config.feature_overlay_kind
        self._feature_scalar_name = "feature"
        self._lcc_vmin = 0.0
        self._lcc_vmax = 100.0
        self._t2m_vmin = 260.0
        self._t2m_vmax = 290.0
        self._wind_vmin = 0.0
        self._wind_vmax = 10.0
        self.feature_vmin = 0.0
        self.feature_vmax = 100.0
        self.feature_label = "LCC (%)"
        self.feature_color_mode = "cloud"
        if self._is_feature_overlay_scene:
            self._init_feature_ranges()
            if self._feature_kind == "lcc":
                self.feature_vmin, self.feature_vmax = self._lcc_vmin, self._lcc_vmax
                self.feature_label = "LCC (%)"
                self.feature_color_mode = "cloud"
            elif self._feature_kind == "t2m":
                self.feature_vmin, self.feature_vmax = self._t2m_vmin, self._t2m_vmax
                self.feature_label = "T2m (K)"
                self.feature_color_mode = "t2m"
            elif self._feature_kind == "wind":
                self.feature_vmin, self.feature_vmax = self._wind_vmin, self._wind_vmax
                self.feature_label = "Wind speed (m/s)"
                self.feature_color_mode = "wind"
            else:
                raise ValueError(
                    f"feature_overlay_kind must be lcc, t2m, or wind; got {self._feature_kind!r}"
                )

        self._terrain_pd = (
            self._build_feature_terrain_polydata(0.0)
            if self._is_feature_overlay_scene
            else build_terrain_surface(grid, self.config.vert_exag)
        )

        if self._is_feature_overlay_scene:
            self.config.show_ridge_highlight = False
            self._compare_bundle = None
            self._slab_pd = vtk.vtkPolyData()
            self._predicted_deck_pd = self._build_predicted_deck_polydata(0.0)
            self._ridge_pd = None
            self.ridge_actor = None
        elif self._is_inversion_scene:
            if self._is_compare_scene:
                self.config.show_ridge_highlight = False
                self._compare_bundle = self._build_compare_deck_polydata(0.0)
                self._slab_pd = self._compare_bundle["actual"]
                self._predicted_deck_pd = self._compare_bundle["predicted"]
                self._ridge_pd = None
                self.ridge_actor = None
            else:
                self._slab_pd = self._build_cloud_deck_polydata(0)
                self._ridge_pd = self._build_ridge_polydata(0)
                self._compare_bundle = None
                self._predicted_deck_pd = None
        else:
            self._slab_pd = self._build_slab_polydata(self.overlay_series[0])
            self._ridge_pd = None
            self._compare_bundle = None
            self._predicted_deck_pd = None

        self.terrain_actor = self._make_terrain_actor()
        self.actual_deck_actor: vtk.vtkActor | None = None
        self.predicted_deck_actor: vtk.vtkActor | None = None
        self.wind_glyph_actor: vtk.vtkActor | None = None
        if self._is_feature_overlay_scene:
            self.predicted_deck_actor = self._make_predicted_deck_actor()
            self.slab_actor = self.predicted_deck_actor
            self.ridge_actor = None
            if self._feature_kind == "wind" and self.grid.u_wind is not None and self.grid.v_wind is not None:
                self.wind_glyph_actor = self._make_wind_glyph_actor(
                    self.grid.u_wind[0],
                    self.grid.v_wind[0],
                )
        elif self._is_inversion_scene:
            if self._is_compare_scene:
                self.actual_deck_actor = self._make_compare_deck_actor(predicted=False)
                self.predicted_deck_actor = self._make_compare_deck_actor(predicted=True)
                self.slab_actor = self.actual_deck_actor
            else:
                self.slab_actor = self._make_cloud_deck_actor()
                self.ridge_actor = (
                    self._make_ridge_actor()
                    if self.config.show_ridge_highlight
                    else None
                )
        else:
            self.slab_actor = self._make_slab_actor()
            self.ridge_actor = None

        if not self._is_feature_overlay_scene and grid.slab_mode == "wind":
            self.wind_glyph_actor = self._make_wind_glyph_actor(
                grid.u_wind[0],
                grid.v_wind[0],
            )

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*BG_COLOR)
        # Compare: actual deck, predicted deck, terrain.
        if self._is_feature_overlay_scene:
            self.renderer.AddActor(self.terrain_actor)
            if self.wind_glyph_actor is not None:
                self.renderer.AddActor(self.wind_glyph_actor)
            if self.predicted_deck_actor is not None:
                self.renderer.AddActor(self.predicted_deck_actor)
            self.renderer.SetUseDepthPeeling(1)
            self.renderer.SetMaximumNumberOfPeels(8)
            self.renderer.SetOcclusionRatio(0.1)
        elif self._is_compare_scene:
            if self.actual_deck_actor is not None:
                self.renderer.AddActor(self.actual_deck_actor)
            if self.predicted_deck_actor is not None:
                self.renderer.AddActor(self.predicted_deck_actor)
            self.renderer.AddActor(self.terrain_actor)
        elif self._is_inversion_scene and self.config.show_cloud_deck:
            self.renderer.AddActor(self.slab_actor)
            self.renderer.AddActor(self.terrain_actor)
        else:
            self.renderer.AddActor(self.terrain_actor)
            if not self._is_inversion_scene:
                self.renderer.AddActor(self.slab_actor)
        if self.ridge_actor is not None:
            self.renderer.AddActor(self.ridge_actor)
        if self._is_compare_scene:
            self.renderer.SetUseDepthPeeling(1)
            self.renderer.SetMaximumNumberOfPeels(8)
            self.renderer.SetOcclusionRatio(0.1)
        if self.wind_glyph_actor is not None and not self._is_feature_overlay_scene:
            self.renderer.AddActor(self.wind_glyph_actor)
        if self.config.show_scalar_bars:
            self._add_scalar_bars()
        self._add_lights()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetMultiSamples(0)

    def _init_slab_range(self) -> None:
        if self.config.slab_mode in ("inversion", "inversion_compare", "feature_overlay"):
            from lib.inversion import cover_to_percent
            deck = self.grid.clt if self.grid.clt is not None else self.grid.cloud_deck_cover
            if deck is not None:
                pct = cover_to_percent(deck[np.isfinite(deck)])
                valid = pct[np.isfinite(pct)]
            else:
                valid = self.overlay_series[np.isfinite(self.overlay_series)]
            if valid.size == 0:
                self.config.slab_vmin, self.config.slab_vmax = 5.0, 95.0
            else:
                self.config.slab_vmin = float(np.nanpercentile(valid, 5))
                self.config.slab_vmax = float(max(95.0, float(np.nanpercentile(valid, 95))))
            return

        valid = self.overlay_series[np.isfinite(self.overlay_series)]
        if valid.size == 0:
            if self.config.slab_mode == "wind":
                self.config.slab_vmin, self.config.slab_vmax = 0.0, 10.0
            elif self.config.slab_mode == "pressure":
                self.config.slab_vmin, self.config.slab_vmax = 400.0, 2500.0
            else:
                self.config.slab_vmin, self.config.slab_vmax = 0.0, 100.0
            return

        if self.config.slab_vmin is None or self.config.slab_vmax is None:
            vmin = float(np.nanpercentile(valid, 5))
            vmax = float(np.nanpercentile(valid, 95))
            if self.config.slab_mode == "wind":
                vmax = max(vmax, vmin + 1.0)
            elif self.config.slab_mode == "pressure":
                vmin, vmax = vmin, max(vmax, vmin + 50.0)
            else:
                if vmin >= vmax:
                    vmin, vmax = 0.0, 100.0
            self.config.slab_vmin = vmin
            self.config.slab_vmax = vmax

        if self.config.slab_vmin >= self.config.slab_vmax:
            if self.config.slab_mode == "wind":
                self.config.slab_vmin, self.config.slab_vmax = 0.0, 10.0
            elif self.config.slab_mode == "pressure":
                self.config.slab_vmin, self.config.slab_vmax = 400.0, 2500.0
            else:
                self.config.slab_vmin, self.config.slab_vmax = 0.0, 100.0

    def _prepare_slab_values(self, values_2d: np.ndarray) -> np.ndarray:
        out = values_2d.astype(np.float32).copy()
        if self.config.slab_mode in ("clt", "pressure"):
            invalid = ~np.isfinite(out)
            out[invalid] = self.config.slab_vmin - 1.0
        return out

    def _build_slab_polydata(self, values_2d: np.ndarray) -> vtk.vtkPolyData:
        return build_scalar_slab(
            self.grid,
            self._prepare_slab_values(values_2d),
            vert_exag=self.config.vert_exag,
            slab_height=self.config.slab_height,
            scalar_name=self._scalar_name,
        )

    def _cloud_deck_cover_at(self, time_idx: int) -> np.ndarray:
        from lib.inversion import cover_to_percent

        if self.grid.clt is not None:
            cover = self.grid.clt[time_idx] if self.grid.clt.ndim == 3 else self.grid.clt
        elif self.grid.cloud_deck_cover is not None:
            cover = (
                self.grid.cloud_deck_cover[time_idx]
                if self.grid.cloud_deck_cover.ndim == 3
                else self.grid.cloud_deck_cover
            )
        else:
            raise ValueError("clt or cloud_deck_cover required for inversion scene")
        return cover_to_percent(cover)

    def _cloud_base_at(self, time_idx: int) -> np.ndarray | None:
        base = self.grid.cloud_base_m
        if base is None:
            return None
        if base.ndim == 3:
            return base[time_idx]
        return base

    def _build_cloud_deck_polydata(self, time_idx: int) -> vtk.vtkPolyData:
        cover = self._cloud_deck_cover_at(time_idx).astype(np.float32)
        base = self._cloud_base_at(time_idx)
        if base is None:
            raise ValueError("cloud_base_m required for inversion cloud layer")
        return self._build_cloud_deck_from_fields(cover, base)

    def _build_ridge_polydata(self, time_idx: int) -> vtk.vtkPolyData:
        return self._build_ridge_from_field(self.overlay_series[time_idx])

    def _make_terrain_actor(self) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._terrain_pd)
        mapper.SetScalarModeToUsePointData()
        if self._is_feature_overlay_scene:
            mapper.SelectColorArray(self._feature_scalar_name)
            mapper.SetLookupTable(_opaque_feature_lut(
                self.feature_vmin,
                self.feature_vmax,
                mode=self._feature_kind,
            ))
            mapper.SetScalarRange(self.feature_vmin, self.feature_vmax)
            mapper.SetColorModeToMapScalars()
        else:
            mapper.SelectColorArray("elevation")
            mapper.SetLookupTable(_terrain_lut(
                self.grid.z[np.isfinite(self.grid.z)],
            ))
            mapper.SetScalarRange(self.elev_vmin, self.elev_vmax)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        if self._is_feature_overlay_scene:
            prop.SetInterpolationToFlat()
            prop.SetAmbient(0.35)
            prop.SetDiffuse(0.85)
            prop.SetSpecular(0.05)
        else:
            prop.SetInterpolationToPhong()
            prop.SetAmbient(0.25)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.15)
            prop.SetSpecularPower(20.0)
        if self._is_compare_scene:
            prop.SetOpacity(self.config.terrain_opacity_compare)
        return actor

    def _init_feature_ranges(self) -> None:
        from lib.meteo_utils import wind_speed

        self._lcc_vmin, self._lcc_vmax = 0.0, 100.0

        if self.grid.feature_t2m is not None:
            valid = self.grid.feature_t2m[np.isfinite(self.grid.feature_t2m)]
            if valid.size:
                lo = float(np.nanpercentile(valid, 2))
                hi = float(np.nanpercentile(valid, 98))
                self._t2m_vmin = math.floor(lo * 2.0) / 2.0
                self._t2m_vmax = math.ceil(hi * 2.0) / 2.0
                if self._t2m_vmax <= self._t2m_vmin:
                    self._t2m_vmax = self._t2m_vmin + 2.0

        if self.grid.u_wind is not None and self.grid.v_wind is not None:
            speed = wind_speed(self.grid.u_wind, self.grid.v_wind)
            valid = speed[np.isfinite(speed)]
            if valid.size:
                self._wind_vmin = 0.0
                self._wind_vmax = max(2.0, math.ceil(float(np.nanpercentile(valid, 98)) * 2.0) / 2.0)

    def _feature_field_at(self, time_idx: float) -> np.ndarray:
        from lib.inversion import cover_to_percent
        from lib.meteo_utils import wind_speed
        from lib.temporal_interp import lerp_time_series

        if self._feature_kind == "lcc":
            if self.grid.cloud_deck_cover is None:
                raise ValueError("cloud_deck_cover required for LCC feature overlay")
            return cover_to_percent(
                lerp_time_series(self.grid.cloud_deck_cover, time_idx),
            ).astype(np.float32)
        if self._feature_kind == "t2m":
            if self.grid.feature_t2m is None:
                raise ValueError("feature_t2m required for T2m feature overlay")
            return lerp_time_series(self.grid.feature_t2m, time_idx).astype(np.float32)
        if self.grid.u_wind is None or self.grid.v_wind is None:
            raise ValueError("u_wind/v_wind required for wind feature overlay")
        u = lerp_time_series(self.grid.u_wind, time_idx)
        v = lerp_time_series(self.grid.v_wind, time_idx)
        return wind_speed(u, v).astype(np.float32)

    def _build_feature_terrain_polydata(self, time_idx: float) -> vtk.vtkPolyData:
        return build_feature_terrain_surface(
            self.grid,
            self._feature_field_at(time_idx),
            vert_exag=self.config.vert_exag,
            scalar_name=self._feature_scalar_name,
        )

    def _build_predicted_deck_polydata(self, time_idx: float) -> vtk.vtkPolyData:
        from lib.terrain_mesh import build_uniform_deck_surface

        cover = self._compare_cover_at(time_idx)
        _actual, predicted_msl = self._deck_heights_at(time_idx)
        return build_uniform_deck_surface(
            self.grid,
            predicted_msl,
            cover,
            vert_exag=self.config.vert_exag,
        )

    def _make_predicted_deck_actor(self) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._predicted_deck_pd)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("deck")
        mapper.SetLookupTable(_deck_surface_lut(0.98, 0.58, 0.12, alpha=self.config.compare_deck_opacity))
        mapper.SetScalarRange(0.0, 1.0)
        mapper.SetColorModeToMapScalars()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToFlat()
        prop.BackfaceCullingOff()
        return actor

    def _make_slab_actor(self) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._slab_pd)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray(self._scalar_name)
        mapper.SetLookupTable(_slab_lut(
            self.config.slab_mode,
            self.config.slab_vmin,
            self.config.slab_vmax,
        ))
        mapper.SetScalarRange(self.config.slab_vmin, self.config.slab_vmax)
        mapper.SetColorModeToMapScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetInterpolationToFlat()
        return actor

    def _make_cloud_deck_actor(self) -> vtk.vtkActor:
        actor = self._make_slab_actor()
        actor.GetMapper().SelectColorArray("cloud_deck")
        max_alpha = self.config.cloud_deck_max_alpha
        if self._is_compare_scene:
            max_alpha = min(max_alpha, 0.32)
        actor.GetMapper().SetLookupTable(_inversion_fog_lut(
            self.config.slab_vmin,
            self.config.slab_vmax,
            max_alpha=max_alpha,
        ))
        actor.GetMapper().SetScalarRange(self.config.slab_vmin, self.config.slab_vmax)
        prop = actor.GetProperty()
        prop.SetInterpolationToFlat()
        prop.BackfaceCullingOff()
        return actor

    def _make_compare_deck_actor(self, *, predicted: bool) -> vtk.vtkActor:
        pd = self._predicted_deck_pd if predicted else self._slab_pd
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("deck")
        alpha = self.config.compare_deck_opacity
        if predicted:
            lut = _deck_surface_lut(0.98, 0.58, 0.12, alpha=alpha)
        else:
            lut = _deck_surface_lut(0.18, 0.78, 0.95, alpha=alpha)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0.0, 1.0)
        mapper.SetColorModeToMapScalars()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToFlat()
        prop.BackfaceCullingOff()
        return actor

    def _make_gap_actor(self) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        pd = self._gap_pd if self._gap_pd is not None else vtk.vtkPolyData()
        mapper.SetInputData(pd)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("gap_class")
        mapper.SetLookupTable(_deck_gap_lut())
        mapper.SetScalarRange(0.0, 3.0)
        mapper.SetColorModeToMapScalars()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToFlat()
        prop.BackfaceCullingOff()
        if pd.GetNumberOfPoints() == 0:
            actor.SetVisibility(0)
        return actor

    def _make_ridge_actor(self) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._ridge_pd)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("inversion")
        mapper.SetLookupTable(_inversion_ridge_lut())
        mapper.SetScalarRange(0.0, 1.0)
        mapper.SetColorModeToMapScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToFlat()
        prop.SetAmbient(0.4)
        prop.SetDiffuse(0.85)
        return actor

    def _wind_glyph_scale(self) -> float:
        span = patch_horizontal_span_m(self.grid)
        if self._is_feature_overlay_scene and self._feature_kind == "wind":
            vmax = max(self._wind_vmax, 0.5)
        else:
            vmax = max(self.config.slab_vmax, 0.5)
        return max(span * 0.018 / vmax, 80.0)

    def _wind_slab_height(self) -> float:
        return self.config.slab_height

    def _wind_glyph_z(self, z_val: float, vert_exag: float) -> float:
        if self._is_feature_overlay_scene:
            return z_val * vert_exag + 120.0 * vert_exag
        z_offset = self._wind_slab_height() * vert_exag
        return z_val * vert_exag + z_offset + 120.0 * vert_exag

    def _make_wind_glyph_actor(self, u_2d: np.ndarray, v_2d: np.ndarray) -> vtk.vtkActor:
        glyph_pd = self._build_wind_glyphs(u_2d, v_2d)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(glyph_pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.95, 0.95, 1.0)
        prop.SetOpacity(0.92)
        return actor

    def _build_wind_glyphs(self, u_2d: np.ndarray, v_2d: np.ndarray) -> vtk.vtkPolyData:
        stride = max(1, self.config.wind_glyph_stride)
        x, y = lonlat_to_local_meters(
            self.grid.lon, self.grid.lat, self.grid.center_lon, self.grid.center_lat,
        )
        ve = self.config.vert_exag

        points = vtk.vtkPoints()
        vectors = vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)
        vectors.SetName("wind")

        n_lat, n_lon = self.grid.z.shape
        for i in range(0, n_lat, stride):
            for j in range(0, n_lon, stride):
                u_val = float(u_2d[i, j])
                v_val = float(v_2d[i, j])
                if not np.isfinite(u_val) or not np.isfinite(v_val):
                    continue
                speed = math.hypot(u_val, v_val)
                if speed < 0.25:
                    continue
                z_val = self.grid.z[i, j]
                if not np.isfinite(z_val):
                    continue
                z_plot = self._wind_glyph_z(z_val, ve)
                points.InsertNextPoint(float(x[j]), float(y[i]), float(z_plot))
                vectors.InsertNextTuple3(u_val, v_val, 0.0)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.GetPointData().SetVectors(vectors)

        arrow = vtk.vtkArrowSource()
        arrow.SetTipLength(0.35)
        arrow.SetTipRadius(0.12)
        arrow.SetShaftRadius(0.045)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(self._wind_glyph_scale())
        glyph.OrientOn()
        glyph.Update()
        return glyph.GetOutput()

    def _add_scalar_bars(self) -> None:
        t_bar = vtk.vtkScalarBarActor()
        t_bar.SetLookupTable(self.terrain_actor.GetMapper().GetLookupTable())
        t_bar.SetTitle("Elevation (m)")
        t_bar.SetNumberOfLabels(4)
        t_bar.SetOrientationToVertical()
        t_bar.SetWidth(0.07)
        t_bar.SetHeight(0.32)
        t_bar.SetPosition(0.86, 0.12)
        t_bar.GetTitleTextProperty().SetColor(0.92, 0.94, 0.97)
        t_bar.GetTitleTextProperty().SetFontSize(12)
        t_bar.GetLabelTextProperty().SetColor(0.8, 0.84, 0.9)
        t_bar.GetLabelTextProperty().SetFontSize(10)
        self.renderer.AddActor2D(t_bar)

        s_bar = vtk.vtkScalarBarActor()
        s_bar.SetLookupTable(self.slab_actor.GetMapper().GetLookupTable())
        s_bar.SetTitle(_slab_bar_title(self.config.slab_mode))
        s_bar.SetNumberOfLabels(4)
        s_bar.SetOrientationToVertical()
        s_bar.SetWidth(0.07)
        s_bar.SetHeight(0.32)
        s_bar.SetPosition(0.93, 0.12)
        s_bar.GetTitleTextProperty().SetColor(0.92, 0.94, 0.97)
        s_bar.GetTitleTextProperty().SetFontSize(12)
        s_bar.GetLabelTextProperty().SetColor(0.8, 0.84, 0.9)
        s_bar.GetLabelTextProperty().SetFontSize(10)
        self.renderer.AddActor2D(s_bar)

    def _add_lights(self) -> None:
        self.renderer.RemoveAllLights()
        cx, cy, cz = self._centroid
        span = patch_horizontal_span_m(self.grid)
        dist = max(span * 0.6, 3000.0)

        key = vtk.vtkLight()
        key.SetLightTypeToSceneLight()
        key.SetPosition(cx - dist, cy - dist, cz + dist * 0.8)
        key.SetFocalPoint(cx, cy, cz)
        key.SetColor(1.0, 0.98, 0.95)
        key.SetIntensity(0.9)
        self.renderer.AddLight(key)

        fill = vtk.vtkLight()
        fill.SetLightTypeToSceneLight()
        fill.SetPosition(cx + dist * 0.6, cy + dist * 0.4, cz + dist * 0.5)
        fill.SetFocalPoint(cx, cy, cz)
        fill.SetColor(0.7, 0.8, 1.0)
        fill.SetIntensity(0.35)
        self.renderer.AddLight(fill)

    def update_frame(self, time_idx: int) -> None:
        self.update_frame_interp(float(time_idx))

    def update_frame_interp(self, time_idx: float) -> None:
        from lib.inversion import cover_to_percent
        from lib.temporal_interp import lerp_time_series

        if self._is_feature_overlay_scene:
            self._terrain_pd.ShallowCopy(self._build_feature_terrain_polydata(time_idx))
            self.terrain_actor.GetMapper().SetInputData(self._terrain_pd)
            self._predicted_deck_pd.ShallowCopy(self._build_predicted_deck_polydata(time_idx))
            self.predicted_deck_actor.GetMapper().SetInputData(self._predicted_deck_pd)
            if self.wind_glyph_actor is not None and self.grid.u_wind is not None:
                u = lerp_time_series(self.grid.u_wind, time_idx)
                v = lerp_time_series(self.grid.v_wind, time_idx)
                glyph_pd = self._build_wind_glyphs(u, v)
                self.wind_glyph_actor.GetMapper().SetInputData(glyph_pd)
        elif self._is_compare_scene:
            bundle = self._build_compare_deck_polydata(time_idx)
            self._slab_pd.ShallowCopy(bundle["actual"])
            self.actual_deck_actor.GetMapper().SetInputData(self._slab_pd)
            self._predicted_deck_pd.ShallowCopy(bundle["predicted"])
            self.predicted_deck_actor.GetMapper().SetInputData(self._predicted_deck_pd)
        elif self._is_inversion_scene:
            if self.config.show_cloud_deck:
                clt = self.grid.clt
                base = self.grid.cloud_base_m
                if clt is None or base is None:
                    raise ValueError("clt and cloud_base_m required for inversion scene")
                cover = cover_to_percent(lerp_time_series(clt, time_idx))
                cbh = lerp_time_series(base, time_idx)
                deck = self._build_cloud_deck_from_fields(cover, cbh)
                self._slab_pd.ShallowCopy(deck)
                self.slab_actor.GetMapper().SetInputData(self._slab_pd)
                self._slab_pd.Modified()
            if self.ridge_actor is not None:
                inv = lerp_time_series(self.overlay_series, time_idx)
                ridge = self._build_ridge_from_field(inv)
                self._ridge_pd.ShallowCopy(ridge)
                self.ridge_actor.GetMapper().SetInputData(self._ridge_pd)
                self._ridge_pd.Modified()
        else:
            values = lerp_time_series(self.overlay_series, time_idx)
            slab = self._build_slab_polydata(values)
            self._slab_pd.ShallowCopy(slab)
            self.slab_actor.GetMapper().SetInputData(self._slab_pd)
            self._slab_pd.Modified()

        if (
            self.wind_glyph_actor is not None
            and self.grid.u_wind is not None
            and not self._is_feature_overlay_scene
        ):
            u = lerp_time_series(self.grid.u_wind, time_idx)
            v = lerp_time_series(self.grid.v_wind, time_idx)
            glyph_pd = self._build_wind_glyphs(u, v)
            self.wind_glyph_actor.GetMapper().SetInputData(glyph_pd)

    def _compare_cover_at(self, time_idx: float) -> np.ndarray:
        from lib.inversion import cover_to_percent
        from lib.temporal_interp import lerp_time_series

        if self.grid.cloud_deck_cover is not None:
            cover = lerp_time_series(self.grid.cloud_deck_cover, time_idx)
        elif self.grid.clt is not None:
            cover = lerp_time_series(self.grid.clt, time_idx)
        else:
            raise ValueError("cloud_deck_cover or clt required for compare scene")
        return cover_to_percent(cover)

    def _build_compare_deck_polydata(self, time_idx: float) -> dict:
        from lib.terrain_mesh import build_uniform_deck_surface

        cover = self._compare_cover_at(time_idx)
        actual_msl, predicted_msl = self._deck_heights_at(time_idx)

        actual_pd = build_uniform_deck_surface(
            self.grid,
            actual_msl,
            cover,
            vert_exag=self.config.vert_exag,
        )
        predicted_pd = build_uniform_deck_surface(
            self.grid,
            predicted_msl,
            cover,
            vert_exag=self.config.vert_exag,
        )

        self._last_actual_deck_msl = actual_msl
        self._last_predicted_deck_msl = predicted_msl

        return {"actual": actual_pd, "predicted": predicted_pd}

    def _deck_heights_at(self, time_idx: float) -> tuple[float, float]:
        from lib.inversion import cover_to_percent, inversion_fog_deck_msl, predicted_fog_deck_msl
        from lib.temporal_interp import lerp_time_series

        if self.grid.cloud_base_m is None:
            return 0.0, 0.0

        cover = self._compare_cover_at(time_idx)
        cbh = lerp_time_series(self.grid.cloud_base_m, time_idx)
        actual = inversion_fog_deck_msl(self.grid.z, cbh, cover)

        pred_mask = None
        actual_mask = None
        if self.grid.predicted_inversion is not None:
            pred_mask = lerp_time_series(self.grid.predicted_inversion, time_idx)
        if self.grid.inversion is not None:
            actual_mask = lerp_time_series(self.grid.inversion, time_idx)

        if pred_mask is None:
            predicted = actual
        else:
            predicted = predicted_fog_deck_msl(
                self.grid.z,
                cbh,
                cover,
                pred_mask,
                actual_mask if actual_mask is not None else None,
            )
        return actual, predicted

    def _build_cloud_deck_from_fields(
        self,
        cover_2d: np.ndarray,
        cloud_base_2d: np.ndarray,
    ) -> vtk.vtkPolyData:
        from lib.terrain_mesh import build_inversion_cloud_layer

        return build_inversion_cloud_layer(
            self.grid,
            cover_2d.astype(np.float32),
            cloud_base_2d,
            vert_exag=self.config.vert_exag,
            uniform_deck=True,
        )

    def _build_ridge_from_field(self, inversion_2d: np.ndarray) -> vtk.vtkPolyData:
        from lib.terrain_mesh import build_inversion_terrain_overlay

        scalar_name = "agreement" if self._is_compare_scene else "inversion"
        return build_inversion_terrain_overlay(
            self.grid,
            inversion_2d,
            vert_exag=self.config.vert_exag,
            scalar_name=scalar_name,
        )

    def set_camera_orbit(
        self,
        frame: int,
        n_frames: int,
        *,
        mode: str = "orbit",
    ) -> None:
        cam = self.renderer.GetActiveCamera()
        cx, cy, cz = self._centroid

        if mode == "static" or n_frames <= 1:
            az = math.radians(self.config.orbit_start_deg)
        else:
            frac = frame / max(n_frames - 1, 1)
            az = math.radians(
                self.config.orbit_start_deg + self.config.orbit_deg * frac,
            )

        r = self._orbit_radius
        if self.config.camera_view == "horizon":
            cam.SetPosition(
                cx + r * math.cos(az),
                cy + r * math.sin(az),
                self._camera_z,
            )
            cam.SetFocalPoint(cx, cy, self._focal_z)
        else:
            el = math.radians(self.config.elevation_angle_deg)
            dx = r * math.cos(el) * math.cos(az)
            dy = r * math.cos(el) * math.sin(az)
            dz = r * math.sin(el)
            cam.SetFocalPoint(cx, cy, cz)
            cam.SetPosition(cx + dx, cy + dy, cz + dz)

        cam.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()

    def configure_render_window(
        self,
        width: int,
        height: int,
        *,
        offscreen: bool = True,
    ) -> None:
        self.render_window.SetSize(width, height)
        self.render_window.SetOffScreenRendering(1 if offscreen else 0)

    def render(self) -> None:
        self.render_window.Render()

    def screenshot_png(self, path: str) -> None:
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.render_window)
        w2i.SetScale(1)
        w2i.ReadFrontBufferOff()
        w2i.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()

    @property
    def cloud_actor(self) -> vtk.vtkActor:
        """Backward-compatible alias for the scalar slab actor."""
        return self.slab_actor

    @property
    def clt_series(self) -> np.ndarray:
        """Backward-compatible alias for overlay time series."""
        return self.overlay_series

"""VTK scene builders for 3D Bavarian Alps terrain + pseudo-3D cloud slab."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import vtk
from vtk.util import numpy_support

from lib.alps_region import alps_mask, scene_centroid, tight_extent

# Ensure OpenGL backend is registered before rendering
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401

BG_COLOR = (0.027, 0.043, 0.078)


def _structured_surface(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    vert_exag: float,
    z_offset: float = 0.0,
) -> vtk.vtkStructuredGrid:
    """Build a single-layer structured grid in lon/lat/Z coordinates."""
    n_lat, n_lon = z.shape
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(n_lon, n_lat, 1)

    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(n_lon * n_lat)
    elev = np.empty(n_lon * n_lat, dtype=np.float32)

    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            z_val = z[i, j]
            if np.isfinite(z_val):
                z_plot = z_val * vert_exag + z_offset
                elev_val = z_val
            else:
                z_plot = 0.0
                elev_val = np.nan
            pts.SetPoint(idx, float(lon[j]), float(lat[i]), float(z_plot))
            elev[idx] = elev_val
            idx += 1

    grid.SetPoints(pts)
    elev_arr = numpy_support.numpy_to_vtk(elev, deep=True)
    elev_arr.SetName("elevation")
    grid.GetPointData().SetScalars(elev_arr)
    return grid


def _threshold_surface(
    grid: vtk.vtkStructuredGrid,
    elev_min: float,
) -> vtk.vtkPolyData:
    """Keep cells whose corner elevations meet the Alps mask threshold."""
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(grid)
    thresh.SetInputArrayToProcess(
        0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
        "elevation",
    )
    thresh.SetLowerThreshold(elev_min)
    thresh.SetUpperThreshold(1.0e30)
    thresh.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)

    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(thresh.GetOutputPort())
    geom.Update()
    return geom.GetOutput()


def _terrain_lut(z_valid: np.ndarray) -> vtk.vtkLookupTable:
    vmin = float(np.nanpercentile(z_valid, 8))
    vmax = float(np.nanpercentile(z_valid, 99.5))
    if vmin >= vmax:
        vmin, vmax = 0.0, 2500.0

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(vmin, vmax)
    for i in range(256):
        t = i / 255.0
        # terrain-like: low green/brown -> high gray/white
        if t < 0.35:
            r, g, b = 0.25 + 0.35 * t, 0.45 + 0.25 * t, 0.18 + 0.1 * t
        elif t < 0.7:
            u = (t - 0.35) / 0.35
            r, g, b = 0.55 + 0.15 * u, 0.52 + 0.08 * u, 0.32 + 0.12 * u
        else:
            u = (t - 0.7) / 0.3
            r, g, b = 0.7 + 0.25 * u, 0.6 + 0.3 * u, 0.44 + 0.5 * u
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.Build()
    return lut


def _cloud_lut(clt_vmin: float, clt_vmax: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetRange(clt_vmin, clt_vmax)
    for i in range(256):
        t = i / 255.0
        # clear -> transparent; cloudy -> white-blue opaque
        alpha = max(0.0, min(1.0, (t - 0.15) / 0.85)) * 0.85
        r = 0.75 + 0.25 * t
        g = 0.82 + 0.15 * t
        b = 0.95
        lut.SetTableValue(i, r, g, b, alpha)
    lut.Build()
    return lut


@dataclass
class AlpsSceneConfig:
    elev_min: float = 900.0
    vert_exag: float = 3.5
    slab_height: float = 600.0
    orbit_deg: float = 120.0
    orbit_start_deg: float = -60.0
    elevation_angle_deg: float = 30.0
    clt_vmin: float = 5.0
    clt_vmax: float = 95.0


class Alps3DScene:
    """Terrain + cloud slab VTK scene with orbit camera."""

    def __init__(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        z: np.ndarray,
        clt_series: np.ndarray,
        config: AlpsSceneConfig | None = None,
    ) -> None:
        self.lon = lon
        self.lat = lat
        self.z = z
        self.clt_series = clt_series
        self.config = config or AlpsSceneConfig()
        self.mask = alps_mask(z, self.config.elev_min)

        if not np.any(self.mask):
            raise ValueError(f"No pixels at or above {self.config.elev_min} m in crop.")

        z_valid = z[self.mask]
        masked_clt = clt_series[:, self.mask]
        if self.config.clt_vmin >= self.config.clt_vmax:
            self.config.clt_vmin = float(np.nanpercentile(masked_clt, 5))
            self.config.clt_vmax = float(np.nanpercentile(masked_clt, 95))
        if self.config.clt_vmin >= self.config.clt_vmax:
            self.config.clt_vmin, self.config.clt_vmax = 0.0, 100.0

        self._centroid = scene_centroid(
            lon, lat, z, self.mask, self.config.vert_exag,
        )
        extent = tight_extent(lon, lat, self.mask, pad_deg=0.02)
        self._span_lon = extent[1] - extent[0]
        self._span_lat = extent[3] - extent[2]
        self._orbit_radius = 1.2 * math.hypot(self._span_lon, self._span_lat)

        terrain_grid = _structured_surface(lon, lat, z, self.config.vert_exag)
        self._terrain_pd = _threshold_surface(terrain_grid, self.config.elev_min)

        slab_offset = self.config.slab_height * self.config.vert_exag
        cloud_grid = _structured_surface(lon, lat, z, self.config.vert_exag, z_offset=slab_offset)
        self._cloud_pd = _threshold_surface(cloud_grid, self.config.elev_min)
        self._set_cloud_scalars(clt_series[0])

        self.terrain_actor = self._make_terrain_actor(z_valid)
        self.cloud_actor = self._make_cloud_actor()

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*BG_COLOR)
        self.renderer.AddActor(self.terrain_actor)
        self.renderer.AddActor(self.cloud_actor)
        self._add_lights()

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetMultiSamples(0)

    def _make_terrain_actor(self, z_valid: np.ndarray) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._terrain_pd)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("elevation")
        mapper.SetLookupTable(_terrain_lut(z_valid))
        mapper.SetScalarRange(
            float(np.nanpercentile(z_valid, 8)),
            float(np.nanpercentile(z_valid, 99.5)),
        )

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetSpecular(0.15)
        prop.SetSpecularPower(20.0)
        return actor

    def _make_cloud_actor(self) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self._cloud_pd)
        mapper.SetScalarModeToUsePointData()
        mapper.SelectColorArray("clt")
        lut = _cloud_lut(self.config.clt_vmin, self.config.clt_vmax)
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(self.config.clt_vmin, self.config.clt_vmax)
        mapper.SetColorModeToMapScalars()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToFlat()
        return actor

    def _add_lights(self) -> None:
        self.renderer.RemoveAllLights()
        key = vtk.vtkLight()
        key.SetPosition(-1.0, -1.0, 1.5)
        key.SetFocalPoint(0.0, 0.0, 0.0)
        key.SetColor(1.0, 0.98, 0.95)
        key.SetIntensity(0.9)
        self.renderer.AddLight(key)

        fill = vtk.vtkLight()
        fill.SetPosition(1.0, 0.5, 0.8)
        fill.SetFocalPoint(0.0, 0.0, 0.0)
        fill.SetColor(0.7, 0.8, 1.0)
        fill.SetIntensity(0.35)
        self.renderer.AddLight(fill)

    def _set_cloud_scalars(self, clt: np.ndarray) -> None:
        flat = np.where(self.mask, clt, np.nan).astype(np.float32).ravel()
        arr = numpy_support.numpy_to_vtk(flat, deep=True)
        arr.SetName("clt")
        self._cloud_pd.GetPointData().SetScalars(arr)
        self._cloud_pd.Modified()

    def update_frame(self, time_idx: int) -> None:
        self._set_cloud_scalars(self.clt_series[time_idx])

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

        el = math.radians(self.config.elevation_angle_deg)
        r = self._orbit_radius
        dx = r * math.cos(el) * math.cos(az)
        dy = r * math.cos(el) * math.sin(az)
        dz = r * math.sin(el) + cz * 0.3

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

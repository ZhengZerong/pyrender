"""Wrapper for offscreen rendering.

Author: Matthew Matl
"""
import os
import sys

from .renderer import Renderer
from .constants import RenderFlags


class OffscreenRenderer(object):
    """A wrapper for offscreen rendering.

    Parameters
    ----------
    viewport_width : int
        The width of the main viewport, in pixels.
    viewport_height : int
        The height of the main viewport, in pixels.
    point_size : float
        The size of screen-space points in pixels.

    Note
    ----
    The valid keys for ``render_flags`` are as follows:

    - ``flip_wireframe``: `bool`, If `True`, all objects will have their
      wireframe modes flipped from what their material indicates.
      Defaults to `False`.
    - ``all_wireframe``: `bool`, If `True`, all objects will be rendered
      in wireframe mode. Defaults to `False`.
    - ``all_solid``: `bool`, If `True`, all objects will be rendered in
      solid mode. Defaults to `False`.
    - ``shadows``: `bool`, If `True`, shadows will be rendered.
      Defaults to `False`.
    - ``vertex_normals``: `bool`, If `True`, vertex normals will be
      rendered as blue lines. Defaults to `False`.
    - ``face_normals``: `bool`, If `True`, face normals will be rendered as
      blue lines. Defaults to `False`.
    - ``cull_faces``: `bool`, If `True`, backfaces will be culled.
      Defaults to `True`.
    - ``point_size`` : float, The point size in pixels. Defaults to 1px.
    """

    def __init__(self, viewport_width, viewport_height, point_size=1.0, 
                 render_flags=None,  **kwargs):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.point_size = point_size

        self._default_render_flags = {
            'flip_wireframe': False,
            'all_wireframe': False,
            'all_solid': False,
            'shadows': False,
            'vertex_normals': False,
            'face_normals': False,
            'cull_faces': True,
            'point_size': 1.0,
        }
        self._render_flags = self._default_render_flags.copy()
        if render_flags is not None:
            self._render_flags.update(render_flags)

        for key in kwargs:
            if key in self.render_flags:
                self._render_flags[key] = kwargs[key]

        # TODO MAC OS BUG FOR SHADOWS
        if sys.platform == 'darwin':
            self._render_flags['shadows'] = False

        self._platform = None
        self._renderer = None
        self._create()

    @property
    def viewport_width(self):
        """int : The width of the main viewport, in pixels.
        """
        return self._viewport_width

    @viewport_width.setter
    def viewport_width(self, value):
        self._viewport_width = int(value)

    @property
    def viewport_height(self):
        """int : The height of the main viewport, in pixels.
        """
        return self._viewport_height

    @viewport_height.setter
    def viewport_height(self, value):
        self._viewport_height = int(value)

    @property
    def point_size(self):
        """float : The pixel size of points in point clouds.
        """
        return self._point_size

    @point_size.setter
    def point_size(self, value):
        self._point_size = float(value)

    @property
    def render_flags(self):
        """dict : Flags for controlling the renderer's behavior.

        - ``flip_wireframe``: `bool`, If `True`, all objects will have their
          wireframe modes flipped from what their material indicates.
          Defaults to `False`.
        - ``all_wireframe``: `bool`, If `True`, all objects will be rendered
          in wireframe mode. Defaults to `False`.
        - ``all_solid``: `bool`, If `True`, all objects will be rendered in
          solid mode. Defaults to `False`.
        - ``shadows``: `bool`, If `True`, shadows will be rendered.
          Defaults to `False`.
        - ``vertex_normals``: `bool`, If `True`, vertex normals will be
          rendered as blue lines. Defaults to `False`.
        - ``face_normals``: `bool`, If `True`, face normals will be rendered as
          blue lines. Defaults to `False`.
        - ``cull_faces``: `bool`, If `True`, backfaces will be culled.
          Defaults to `True`.
        - ``point_size`` : float, The point size in pixels. Defaults to 1px.

        """
        return self._render_flags

    @render_flags.setter
    def render_flags(self, value):
        self._render_flags = value

    def render(self, scene, flags=RenderFlags.NONE):
        """Render a scene with the given set of flags.

        Parameters
        ----------
        scene : :class:`Scene`
            A scene to render.
        flags : int
            A bitwise or of one or more flags from :class:`.RenderFlags`.

        Returns
        -------
        color_im : (h, w, 3) uint8 or (h, w, 4) uint8
            The color buffer in RGB format, or in RGBA format if
            :attr:`.RenderFlags.RGBA` is set.
            Not returned if flags includes :attr:`.RenderFlags.DEPTH_ONLY`.
        depth_im : (h, w) float32
            The depth buffer in linear units.
        """
        self._platform.make_current()
        # If platform does not support dynamically-resizing framebuffers,
        # destroy it and restart it
        if (self._platform.viewport_height != self.viewport_height or
                self._platform.viewport_width != self.viewport_width):
            if not self._platform.supports_framebuffers():
                self.delete()
                self._create()

        self._platform.make_current()
        self._renderer.viewport_width = self.viewport_width
        self._renderer.viewport_height = self.viewport_height
        self._renderer.point_size = self.point_size

        if self.render_flags['flip_wireframe']:
            flags |= RenderFlags.FLIP_WIREFRAME
        elif self.render_flags['all_wireframe']:
            flags |= RenderFlags.ALL_WIREFRAME
        elif self.render_flags['all_solid']:
            flags |= RenderFlags.ALL_SOLID
        if self.render_flags['shadows']:
            flags |= RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.SHADOWS_SPOT
        if self.render_flags['vertex_normals']:
            flags |= RenderFlags.VERTEX_NORMALS
        if self.render_flags['face_normals']:
            flags |= RenderFlags.FACE_NORMALS
        if not self.render_flags['cull_faces']:
            flags |= RenderFlags.SKIP_CULL_FACES
        
        if self._platform.supports_framebuffers():
            flags |= RenderFlags.OFFSCREEN
            return self._renderer.render(scene, flags)
        else:
            self._renderer.render(scene, flags)
            depth = self._renderer.read_depth_buf()
            if flags & RenderFlags.DEPTH_ONLY:
                return depth
            color = self._renderer.read_color_buf()
            return color, depth

    def delete(self):
        """Free all OpenGL resources.
        """
        self._renderer.delete()
        self._platform.delete_context()
        self._renderer = None
        self._platform = None

    def _create(self):
        if 'PYOPENGL_PLATFORM' not in os.environ:
            
            from .platforms.pyglet_platform import PygletPlatform
            self._platform = PygletPlatform(self.viewport_width,
                                            self.viewport_height)
        elif os.environ['PYOPENGL_PLATFORM'] == 'egl':
            from .platforms import egl
            device_id = int(os.environ.get('EGL_DEVICE_ID', '0'))
            egl_device = egl.get_device_by_index(device_id)
            self._platform = egl.EGLPlatform(self.viewport_width,
                                             self.viewport_height,
                                             device=egl_device)
        elif os.environ['PYOPENGL_PLATFORM'] == 'osmesa':
            from .platforms.osmesa import OSMesaPlatform
            self._platform = OSMesaPlatform(self.viewport_width,
                                            self.viewport_height)
        else:
            raise ValueError('Unsupported PyOpenGL platform: {}'.format(
                os.environ['PYOPENGL_PLATFORM']
            ))
        self._platform.init_context()
        self._platform.make_current()
        self._renderer = Renderer(self.viewport_width, self.viewport_height)

    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass


__all__ = ['OffscreenRenderer']

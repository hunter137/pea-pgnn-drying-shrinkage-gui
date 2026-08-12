from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tkinter as tk
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pea_pgnn.gui import PEAPGNNApp


@unittest.skipUnless((ROOT / "artifacts" / "deployment" / "manifest.json").is_file(), "deployment artifact not trained")
class GUIInputInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest("Tk display is unavailable: {}".format(exc))
        cls.root.geometry("1180x720")
        cls.app = PEAPGNNApp(cls.root, ROOT / "artifacts" / "deployment")
        cls.root.update()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "root"):
            cls.root.destroy()

    def _wheel_event(self, delta):
        canvas = self.app.input_canvas
        return SimpleNamespace(
            x_root=canvas.winfo_rootx() + max(2, canvas.winfo_width() // 2),
            y_root=canvas.winfo_rooty() + max(2, canvas.winfo_height() // 2),
            delta=delta,
            widget=canvas,
        )

    def test_mouse_wheel_scrolls_input_sheet_in_both_directions(self):
        canvas = self.app.input_canvas
        canvas.yview_moveto(0.0)
        self.root.update_idletasks()
        before = canvas.yview()[0]
        handled = self.app._on_global_mousewheel(self._wheel_event(-120))
        self.root.update_idletasks()
        after_down = canvas.yview()[0]
        self.app._on_global_mousewheel(self._wheel_event(120))
        self.root.update_idletasks()
        after_up = canvas.yview()[0]
        self.assertEqual(handled, "break")
        self.assertGreater(after_down, before)
        self.assertLess(after_up, after_down)

    def test_local_shortcut_reaches_geometry_without_project_tree(self):
        self.app.input_canvas.yview_moveto(0.0)
        self.app._scroll_input_section("geometry")
        self.root.update_idletasks()
        self.assertGreater(self.app.input_canvas.yview()[0], 0.0)


if __name__ == "__main__":
    unittest.main()

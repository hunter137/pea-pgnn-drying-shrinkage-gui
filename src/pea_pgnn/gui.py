"""Three-column desktop interface for the validated shared predictor."""

from __future__ import annotations

import os
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .formula_converter import convert_formula, mathml_to_formula
from .formula_registry import FormulaRegistry
from .inference import Predictor
from .reporting import build_pdf_report, make_report_id
from .mathtype_bridge import (
    MathTypeBridgeError,
    copy_current_formula,
    find_mathtype,
    launch_mathtype,
    paste_current_formula,
    read_mathml_clipboard,
    set_clipboard_text,
)
from .trial_calculation import (
    DEFAULT_TRIAL_CONDITION,
    analyse_formula_curve,
    build_trial_age_grid,
    parse_number_list,
)


COLORS = {
    "background": "#F5F7FA",
    "card": "#FFFFFF",
    "border": "#D5DBE3",
    "blue": "#1557A6",
    "blue_dark": "#124E91",
    "blue_light": "#EAF2FB",
    "text": "#263238",
    "muted": "#667085",
    "green": "#16833A",
    "green_light": "#E8F7EC",
    "orange": "#C86A00",
    "orange_light": "#FFF3E0",
    "red": "#B42318",
    "red_light": "#FDECEA",
}

WORKBENCH = {
    "chrome": "#E6E8EB",
    "chrome_dark": "#D3D7DC",
    "workspace": "#F2F3F5",
    "panel": "#FFFFFF",
    "header": "#ECEFF2",
    "border": "#AEB4BC",
    "text": "#20252B",
    "muted": "#5E6874",
    "accent": "#1F5A94",
    "accent_dark": "#184670",
    "selection": "#D9E8F6",
    "console": "#FAFAFA",
}

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 8)
FONT_CARD = ("Segoe UI Semibold", 11)
FONT_TITLE = ("Segoe UI Semibold", 24)
FONT_SUBTITLE = ("Segoe UI", 11)
FONT_VALUE = ("Segoe UI Semibold", 25)
MATH_FONT = ("Cambria Math", 10)
MICROSTRAIN = "\u00b5\u03b5"


class Card(tk.Frame):
    def __init__(self, parent, title, icon="", **kwargs):
        super().__init__(parent, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1, **kwargs)
        header = tk.Frame(self, bg=COLORS["card"], height=45)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text=(icon + "  " if icon else "") + title,
            font=FONT_CARD,
            fg=COLORS["blue_dark"],
            bg=COLORS["card"],
            anchor="w",
        ).pack(fill="both", expand=True, padx=14)
        tk.Frame(self, bg=COLORS["border"], height=1).pack(fill="x")
        self.body = tk.Frame(self, bg=COLORS["card"])
        self.body.pack(fill="both", expand=True)


class WorkbenchPanel(tk.Frame):
    """Compact desktop-style pane with a fixed tool-window caption."""

    def __init__(self, parent, title, **kwargs):
        super().__init__(
            parent, bg=WORKBENCH["panel"], highlightbackground=WORKBENCH["border"],
            highlightthickness=1, **kwargs,
        )
        caption = tk.Frame(self, bg=WORKBENCH["header"], height=28)
        caption.pack(fill="x")
        caption.pack_propagate(False)
        tk.Label(
            caption, text=title, font=("Segoe UI Semibold", 9), fg=WORKBENCH["text"],
            bg=WORKBENCH["header"], anchor="w",
        ).pack(fill="both", expand=True, padx=8)
        tk.Frame(self, bg=WORKBENCH["border"], height=1).pack(fill="x")
        self.body = tk.Frame(self, bg=WORKBENCH["panel"])
        self.body.pack(fill="both", expand=True)


class PEAPGNNApp:
    def __init__(self, root, artifact_directory):
        self.root = root
        self.predictor = Predictor(artifact_directory)
        self.last_result = None
        self.last_condition = None
        self.fields = {}
        self.status_text = tk.StringVar(value="Ready")
        self.coordinate_text = tk.StringVar(value="No active result")
        self.root.title("PEA-PGNN Drying-Shrinkage Prediction V1.0.0")
        self.root.geometry("1480x920")
        self.root.minsize(1180, 720)
        self.root.configure(bg=WORKBENCH["workspace"])
        self._configure_styles()
        self._build_menu()
        self._build_toolbar()
        self._build_content()
        self._build_footer()
        self._reset_chart()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("PEA.TEntry", font=("Segoe UI", 9), padding=(5, 4), fieldbackground="white", bordercolor=WORKBENCH["border"])
        style.configure("PEA.TCombobox", font=("Segoe UI", 9), padding=(5, 3))
        style.configure("Workbench.TNotebook", background=WORKBENCH["workspace"], borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure(
            "Workbench.TNotebook.Tab", font=("Segoe UI", 9), padding=(13, 6),
            background=WORKBENCH["chrome_dark"], foreground=WORKBENCH["text"],
        )
        style.map(
            "Workbench.TNotebook.Tab",
            background=[("selected", WORKBENCH["panel"])],
            foreground=[("selected", WORKBENCH["accent_dark"])],
        )
        style.configure(
            "Workbench.Treeview", font=("Segoe UI", 9), rowheight=23,
            background="white", fieldbackground="white", foreground=WORKBENCH["text"],
        )
        style.configure("Workbench.Treeview.Heading", font=("Segoe UI Semibold", 8), padding=(4, 4))
        style.map("Workbench.Treeview", background=[("selected", WORKBENCH["selection"])], foreground=[("selected", WORKBENCH["text"])])
        style.configure(
            "Weight.Horizontal.TProgressbar",
            troughcolor="#EDF1F5",
            background=COLORS["blue"],
            lightcolor=COLORS["blue"],
            darkcolor=COLORS["blue"],
            bordercolor="#EDF1F5",
        )

    def _build_menu(self):
        menu = tk.Menu(self.root, tearoff=False, font=("Segoe UI", 9))
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Run Prediction\tF5", command=self.run_prediction, accelerator="F5")
        file_menu.add_separator()
        file_menu.add_command(label="Generate PDF Report...", command=self.export_report)
        file_menu.add_command(label="Export Curve...", command=self.export_curve)
        file_menu.add_command(label="Batch Prediction...", command=self.batch_prediction)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menu.add_cascade(label="File", menu=file_menu)
        model_menu = tk.Menu(menu, tearoff=False)
        model_menu.add_command(label="Reset Input", command=self.reset)
        model_menu.add_command(label="Formula Library...", command=self.open_formula_library)
        model_menu.add_separator()
        model_menu.add_command(label="Open User Formula Space", command=lambda: os.startfile(str(self.predictor.formulas.directory)))
        menu.add_cascade(label="Model", menu=model_menu)
        view_menu = tk.Menu(menu, tearoff=False)
        view_menu.add_command(label="Curve View", command=lambda: self.workspace_tabs.select(self.curve_tab))
        view_menu.add_command(label="Result Table", command=lambda: self.workspace_tabs.select(self.table_tab))
        view_menu.add_command(label="Diagnostics", command=lambda: self.workspace_tabs.select(self.diagnostics_tab))
        menu.add_cascade(label="View", menu=view_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About PEA-PGNN V1.0.0", command=self._show_about)
        menu.add_cascade(label="Help", menu=help_menu)
        self.root.configure(menu=menu)
        self.root.bind("<F5>", lambda event: self.run_prediction())

    def _build_toolbar(self):
        toolbar = tk.Frame(self.root, bg=WORKBENCH["chrome"], height=48, bd=0)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)
        tk.Frame(toolbar, bg=WORKBENCH["border"], height=1).pack(fill="x", side="bottom")
        brand = tk.Frame(toolbar, bg=WORKBENCH["chrome"])
        brand.pack(side="left", fill="y", padx=(8, 16))
        tk.Label(brand, text="PEA-PGNN", font=("Segoe UI Semibold", 11), fg=WORKBENCH["accent_dark"], bg=WORKBENCH["chrome"]).pack(side="left", pady=5)
        tk.Label(brand, text="Drying Shrinkage", font=("Segoe UI", 9), fg=WORKBENCH["muted"], bg=WORKBENCH["chrome"]).pack(side="left", padx=(7, 0), pady=5)
        tk.Frame(toolbar, bg=WORKBENCH["border"], width=1).pack(side="left", fill="y", pady=7)
        self._tool_button(toolbar, "Run", self.run_prediction, primary=True).pack(side="left", padx=(8, 3), pady=7)
        self._tool_button(toolbar, "Reset", self.reset).pack(side="left", padx=3, pady=7)
        self._tool_button(toolbar, "Report", self.export_report).pack(side="left", padx=3, pady=7)
        self._tool_button(toolbar, "Export CSV", self.export_curve).pack(side="left", padx=3, pady=7)
        self._tool_button(toolbar, "Batch", self.batch_prediction).pack(side="left", padx=3, pady=7)
        tk.Frame(toolbar, bg=WORKBENCH["border"], width=1).pack(side="left", fill="y", padx=7, pady=7)
        self._tool_button(toolbar, "Formula Library", self.open_formula_library).pack(side="left", padx=3, pady=7)
        tk.Label(
            toolbar,
            text=self.predictor.model_label + "  |  39 variables  |  104,200 parameters/member",
            font=("Segoe UI", 8), fg=WORKBENCH["muted"], bg=WORKBENCH["chrome"],
        ).pack(side="right", padx=12)

    def _tool_button(self, parent, text, command, primary=False):
        return tk.Button(
            parent, text=text, command=command, font=("Segoe UI Semibold" if primary else "Segoe UI", 9),
            fg=("white" if primary else WORKBENCH["text"]),
            bg=(WORKBENCH["accent"] if primary else WORKBENCH["chrome"]),
            activeforeground=("white" if primary else WORKBENCH["text"]),
            activebackground=(WORKBENCH["accent_dark"] if primary else "#F8F9FA"),
            relief="solid", bd=1, padx=13, pady=4,
        )

    def _show_about(self):
        messagebox.showinfo(
            "About PEA-PGNN",
            "PEA-PGNN Drying-Shrinkage Prediction\nVersion 1.0.0\n\n"
            "Three-member deployment ensemble\nFrozen 39-variable implementation\n"
            "Research software; engineering verification is required.",
            parent=self.root,
        )

    def _build_content(self):
        content = tk.PanedWindow(
            self.root, orient="vertical", sashwidth=5, sashrelief="flat",
            bg=WORKBENCH["chrome_dark"], bd=0,
        )
        content.pack(fill="both", expand=True, padx=5, pady=5)
        work_area = tk.PanedWindow(
            content, orient="horizontal", sashwidth=5, sashrelief="flat",
            bg=WORKBENCH["chrome_dark"], bd=0,
        )
        console = WorkbenchPanel(content, "Messages")
        content.add(work_area, stretch="always", minsize=500)
        content.add(console, height=116, minsize=80, stretch="never")

        left = tk.PanedWindow(
            work_area, orient="vertical", sashwidth=5, sashrelief="flat",
            bg=WORKBENCH["chrome_dark"], bd=0,
        )
        center = tk.Frame(work_area, bg=WORKBENCH["workspace"])
        right = tk.Frame(work_area, bg=WORKBENCH["workspace"])
        work_area.add(left, width=365, minsize=330, stretch="never")
        work_area.add(center, minsize=520, stretch="always")
        work_area.add(right, width=350, minsize=310, stretch="never")

        self._build_project_navigator(left)
        self._build_inputs(left)
        self._build_results(center)
        self._build_interpretation(right)
        self._build_console(console.body)

    def _build_project_navigator(self, parent):
        pane = WorkbenchPanel(parent, "Project")
        parent.add(pane, height=170, minsize=130, stretch="never")
        tree = ttk.Treeview(pane.body, show="tree", style="Workbench.Treeview", selectmode="browse")
        root_item = tree.insert("", "end", text="PEA-PGNN Model", open=True)
        input_item = tree.insert(root_item, "end", text="Input Condition", open=True)
        material_item = tree.insert(input_item, "end", text="Material")
        exposure_item = tree.insert(input_item, "end", text="Exposure & Curing")
        geometry_item = tree.insert(input_item, "end", text="Geometry & Query")
        model_item = tree.insert(root_item, "end", text="Prediction Model", open=True)
        ensemble_item = tree.insert(model_item, "end", text="Deployment Ensemble (3)")
        refs = tree.insert(root_item, "end", text="Reference Equations", open=True)
        b3_item = tree.insert(refs, "end", text="Model B3")
        gl_item = tree.insert(refs, "end", text="GL2000")
        aci_item = tree.insert(refs, "end", text="ACI 209")
        library_item = tree.insert(root_item, "end", text="User Formula Library")
        tree.pack(fill="both", expand=True, padx=3, pady=3)
        tree.selection_set(input_item)
        self.project_tree = tree
        self._project_actions = {
            root_item: lambda: self.workspace_tabs.select(self.curve_tab),
            input_item: lambda: self.workspace_tabs.select(self.curve_tab),
            material_item: lambda: self._scroll_input_section("material"),
            exposure_item: lambda: self._scroll_input_section("exposure"),
            geometry_item: lambda: self._scroll_input_section("geometry"),
            model_item: lambda: self.workspace_tabs.select(self.diagnostics_tab),
            ensemble_item: lambda: self.workspace_tabs.select(self.diagnostics_tab),
            refs: self.open_formula_library,
            b3_item: self.open_formula_library,
            gl_item: self.open_formula_library,
            aci_item: self.open_formula_library,
            library_item: self.open_formula_library,
        }
        tree.bind("<Double-1>", self._activate_project_item)
        tree.bind("<Return>", self._activate_project_item)

    def _activate_project_item(self, event=None):
        selection = self.project_tree.selection()
        if selection:
            action = self._project_actions.get(selection[0])
            if action:
                action()

    def _scroll_input_section(self, section):
        self.workspace_tabs.select(self.curve_tab)
        widget = self.input_section_widgets.get(section)
        if widget is None:
            return
        self.input_canvas.update_idletasks()
        total = max(1, self.input_form.winfo_reqheight())
        self.input_canvas.yview_moveto(max(0.0, min(1.0, widget.winfo_y() / float(total))))
        self.status_text.set("Input properties | {}".format(section.title()))

    def _pointer_is_over_input_panel(self, event):
        """Return whether the physical pointer is inside the input scroller."""
        if not hasattr(self, "input_scroll_host"):
            return False
        host_path = str(self.input_scroll_host)

        def is_inside(widget):
            if widget is None:
                return False
            widget_path = str(widget)
            return widget_path == host_path or widget_path.startswith(host_path + ".")

        # The event target is reliable for ordinary wheel events and remains
        # available if a window manager cannot resolve the physical pointer.
        if is_inside(getattr(event, "widget", None)):
            return True
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
        except (AttributeError, tk.TclError):
            widget = None
        return is_inside(widget)

    def _scroll_input_units(self, units):
        if hasattr(self, "input_canvas"):
            self.input_canvas.yview_scroll(int(units), "units")

    def _on_input_mousewheel(self, event):
        """Route Windows/macOS wheel and precision-touchpad input to the form."""
        delta = int(getattr(event, "delta", 0))
        if delta:
            notches = max(1, abs(delta) // 120)
            direction = -1 if delta > 0 else 1
            self._scroll_input_units(direction * notches * 3)
        return "break"

    def _on_global_mousewheel(self, event):
        # Tk sends the event to the nested entry/label under the pointer, not
        # automatically to the canvas that owns the scrollable property sheet.
        if self._pointer_is_over_input_panel(event):
            return self._on_input_mousewheel(event)
        return None

    def _on_global_mousewheel_button(self, event):
        if not self._pointer_is_over_input_panel(event):
            return None
        self._scroll_input_units(-3 if event.num == 4 else 3)
        return "break"

    def _build_inputs(self, parent):
        card = WorkbenchPanel(parent, "Properties - Input Condition")
        parent.add(card, minsize=340, stretch="always")
        body = card.body

        # Keep the primary workflow visible at every scroll position.  The
        # material/exposure form is intentionally scrollable, but hiding the
        # prediction button below the fold makes the first-run experience
        # unnecessarily ambiguous on smaller or high-DPI screens.
        actions = tk.Frame(body, bg=WORKBENCH["header"])
        actions.pack(side="bottom", fill="x")
        tk.Frame(actions, bg=WORKBENCH["border"], height=1).pack(fill="x")
        tk.Button(
            actions,
            text="Apply and Run (F5)",
            command=self.run_prediction,
            font=("Segoe UI Semibold", 9),
            fg="white",
            bg=WORKBENCH["accent"],
            activebackground=WORKBENCH["accent_dark"],
            activeforeground="white",
            relief="solid", bd=1, cursor="hand2", padx=14, pady=5,
        ).pack(side="right", padx=6, pady=6)
        self._secondary_button(actions, "Reset", self.reset).pack(side="right", padx=(0, 2), pady=6)

        # Put navigation next to the fields. First-time users do not need to
        # discover the separate Project Navigator to reach lower properties.
        section_bar = tk.Frame(body, bg=WORKBENCH["header"])
        section_bar.pack(side="top", fill="x")
        hint_row = tk.Frame(section_bar, bg=WORKBENCH["header"])
        hint_row.pack(fill="x", padx=7, pady=(5, 2))
        tk.Label(
            hint_row, text="INPUT SECTIONS", font=("Segoe UI Semibold", 7),
            fg=WORKBENCH["text"], bg=WORKBENCH["header"], anchor="w",
        ).pack(side="left")
        tk.Label(
            hint_row, text="Mouse wheel scrolls this panel", font=("Segoe UI", 7),
            fg=WORKBENCH["muted"], bg=WORKBENCH["header"], anchor="e",
        ).pack(side="right")
        shortcuts = tk.Frame(section_bar, bg=WORKBENCH["header"])
        shortcuts.pack(fill="x", padx=6, pady=(0, 5))
        for column, (label, section) in enumerate((
            ("Material", "material"),
            ("Exposure", "exposure"),
            ("Geometry / Age", "geometry"),
        )):
            shortcuts.grid_columnconfigure(column, weight=1, uniform="input_section")
            tk.Button(
                shortcuts, text=label,
                command=lambda target=section: self._scroll_input_section(target),
                font=("Segoe UI", 8), fg=WORKBENCH["text"], bg=WORKBENCH["chrome"],
                activebackground="white", activeforeground=WORKBENCH["text"],
                relief="solid", bd=1, cursor="hand2", padx=4, pady=2,
            ).grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 2, 0))
        tk.Frame(section_bar, bg=WORKBENCH["border"], height=1).pack(fill="x")

        scroller = tk.Frame(body, bg=WORKBENCH["panel"])
        scroller.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(
            scroller, bg=WORKBENCH["panel"], highlightthickness=0,
            yscrollincrement=18,
        )
        scrollbar = ttk.Scrollbar(scroller, orient="vertical", command=canvas.yview)
        form = tk.Frame(canvas, bg=WORKBENCH["panel"])
        form.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.input_canvas = canvas
        self.input_form = form
        self.input_scroll_host = scroller
        self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_global_mousewheel_button, add="+")
        self.root.bind_all("<Button-5>", self._on_global_mousewheel_button, add="+")
        self.input_section_widgets = {}
        self.input_section_widgets["material"] = self._section_label(form, "Material")
        self._field(form, "Cement", "cement", 371, "kg/m³")
        self._field(form, "Water", "water", 186, "kg/m³")
        self._field(form, "Aggregate", "aggregate", 1859, "kg/m³")
        self._field(form, "Water-binder ratio", "wb", 0.48, "-", symbol=("w/b", ""))
        self._field(form, "28-d compressive strength", "fc28", 37, "MPa", symbol=("f", "c,28"))
        self._field(form, "28-d elastic modulus", "Ec28", 25958, "MPa", symbol=("E", "c,28"))
        self._combo(form, "Cement type", "cement_type_code", ["1", "2", "3", "4"], "2")
        self._combo(form, "Aggregate type", "agg_type_code", [str(i) for i in range(1, 8)], "1")

        self.input_section_widgets["exposure"] = self._section_label(form, "Exposure and curing")
        self._field(form, "Curing age", "t0", 7, "d", symbol=("t", "0"))
        self._field(form, "Relative humidity", "RH", 50, "%", symbol=("RH", ""))
        self._field(form, "Temperature", "T", 23, "deg C")
        self._combo(form, "Curing type", "curing_type_code", [str(i) for i in range(1, 8)], "1")

        self.input_section_widgets["geometry"] = self._section_label(form, "Geometry and query")
        self._field(form, "Theoretical thickness", "h0", 45.5, "mm", update_derived=True, symbol=("h", "0"))
        self.derived_vs = tk.StringVar(value="22.75 mm")
        row = tk.Frame(form, bg=WORKBENCH["panel"])
        row.pack(fill="x", padx=7, pady=2)
        tk.Label(row, text="Derived V/S", font=("Segoe UI", 8), bg=WORKBENCH["panel"], fg=WORKBENCH["text"], width=21, anchor="w").pack(side="left")
        tk.Label(row, textvariable=self.derived_vs, font=("Segoe UI Semibold", 8), bg=WORKBENCH["panel"], fg=WORKBENCH["accent"], anchor="e").pack(side="right", padx=(0, 8))
        self._combo(form, "Geometry", "geometry", ["Prism", "Cylinder", "Hollow cylinder", "Slab"], "Prism")
        self._field(form, "Query drying age", "query_age", 365, "d", symbol=("t", ""))

    def _section_label(self, parent, text):
        row = tk.Frame(parent, bg=WORKBENCH["header"], height=24)
        row.pack(fill="x", pady=(6, 1))
        row.pack_propagate(False)
        tk.Label(
            row, text=text, font=("Segoe UI Semibold", 8), fg=WORKBENCH["text"],
            bg=WORKBENCH["header"], anchor="w",
        ).pack(fill="both", expand=True, padx=7)
        return row

    def _field(self, parent, label, key, default, unit, update_derived=False, symbol=None):
        row = tk.Frame(parent, bg=WORKBENCH["panel"])
        row.pack(fill="x", padx=7, pady=2)
        row.grid_columnconfigure(1, weight=1)
        label_row = tk.Frame(row, bg=WORKBENCH["panel"], width=150, height=25)
        label_row.grid(row=0, column=0, sticky="w")
        label_row.pack_propagate(False)
        tk.Label(label_row, text=label + (", " if symbol else ""), font=("Segoe UI", 8), bg=WORKBENCH["panel"], fg=WORKBENCH["text"], anchor="w").pack(side="left")
        if symbol:
            base, subscript = symbol
            tk.Label(label_row, text=base, font=("Cambria Math", 9, "italic"), bg=WORKBENCH["panel"], fg=WORKBENCH["text"]).pack(side="left")
            if subscript:
                tk.Label(label_row, text=subscript, font=("Cambria Math", 6, "italic"), bg=WORKBENCH["panel"], fg=WORKBENCH["text"]).pack(side="left", pady=(5, 0))
        value_row = tk.Frame(row, bg=WORKBENCH["panel"])
        value_row.grid(row=0, column=1, sticky="ew")
        variable = tk.StringVar(value=str(default))
        entry = ttk.Entry(value_row, textvariable=variable, style="PEA.TEntry", justify="right")
        entry.pack(side="left", fill="x", expand=True)
        tk.Label(value_row, text=unit, font=("Segoe UI", 7), bg=WORKBENCH["panel"], fg=WORKBENCH["muted"], width=8, anchor="w").pack(side="left", padx=(5, 0))
        self.fields[key] = variable
        if update_derived:
            variable.trace_add("write", lambda *_: self._update_derived_vs())

    def _combo(self, parent, label, key, values, default):
        row = tk.Frame(parent, bg=WORKBENCH["panel"])
        row.pack(fill="x", padx=7, pady=2)
        row.grid_columnconfigure(1, weight=1)
        tk.Label(row, text=label, font=("Segoe UI", 8), bg=WORKBENCH["panel"], fg=WORKBENCH["text"], width=21, anchor="w").grid(row=0, column=0, sticky="w")
        variable = tk.StringVar(value=default)
        combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", style="PEA.TCombobox")
        combo.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        # Scrolling over a categorical control should move the property sheet,
        # not silently change the selected cement/aggregate/curing type.
        combo.bind("<MouseWheel>", self._on_input_mousewheel)
        combo.bind("<Button-4>", lambda event: (self._scroll_input_units(-3), "break")[1])
        combo.bind("<Button-5>", lambda event: (self._scroll_input_units(3), "break")[1])
        self.fields[key] = variable

    def _secondary_button(self, parent, text, command):
        return tk.Button(
            parent, text=text, command=command, font=("Segoe UI", 8),
            fg=WORKBENCH["text"], bg=WORKBENCH["chrome"], activebackground="white",
            relief="solid", bd=1, cursor="hand2", padx=10, pady=4,
        )

    def _build_results(self, parent):
        self.workspace_tabs = ttk.Notebook(parent, style="Workbench.TNotebook")
        self.workspace_tabs.pack(fill="both", expand=True)
        self.curve_tab = tk.Frame(self.workspace_tabs, bg=WORKBENCH["panel"])
        self.table_tab = tk.Frame(self.workspace_tabs, bg=WORKBENCH["panel"])
        self.diagnostics_tab = tk.Frame(self.workspace_tabs, bg=WORKBENCH["panel"])
        self.workspace_tabs.add(self.curve_tab, text="Curve View")
        self.workspace_tabs.add(self.table_tab, text="Key-Age Table")
        self.workspace_tabs.add(self.diagnostics_tab, text="Model Diagnostics")

        body = self.curve_tab
        summary = tk.Frame(body, bg=WORKBENCH["header"], height=105)
        summary.pack(fill="x")
        summary.pack_propagate(False)
        tk.Frame(summary, bg=WORKBENCH["border"], height=1).pack(fill="x", side="bottom")
        result_block = tk.Frame(summary, bg=WORKBENCH["header"], width=305)
        result_block.pack(side="left", fill="y", padx=(8, 0))
        result_block.pack_propagate(False)
        self.result_value = tk.StringVar(value="Run a prediction")
        self.result_age = tk.StringVar(value="Point prediction | nominal interval not reported")
        self.result_figure = Figure(figsize=(3.0, 0.55), dpi=100, facecolor=WORKBENCH["header"])
        self.result_axes = self.result_figure.add_subplot(111)
        self.result_axes.axis("off")
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, master=result_block)
        self.result_canvas.get_tk_widget().pack(fill="x", pady=(8, 0))
        self._draw_result_value()
        tk.Label(result_block, textvariable=self.result_age, font=("Segoe UI", 7), fg=WORKBENCH["muted"], bg=WORKBENCH["header"]).pack()

        reference_row = tk.Frame(summary, bg=WORKBENCH["header"])
        reference_row.pack(side="left", fill="both", expand=True, padx=8, pady=6)
        tk.Label(reference_row, text="REFERENCE EQUATIONS", font=("Segoe UI Semibold", 7), fg=WORKBENCH["muted"], bg=WORKBENCH["header"], anchor="w").pack(fill="x")
        values_row = tk.Frame(reference_row, bg=WORKBENCH["header"])
        values_row.pack(fill="x", pady=(5, 3))
        self.reference_values = {}
        for name in ("Model B3", "GL2000", "ACI 209"):
            panel = tk.Frame(values_row, bg=WORKBENCH["header"])
            panel.pack(side="left", fill="x", expand=True, padx=(0, 8))
            tk.Label(panel, text=name, font=("Segoe UI", 7), fg=WORKBENCH["muted"], bg=WORKBENCH["header"], anchor="w").pack(fill="x")
            variable = tk.StringVar(value="-")
            tk.Label(panel, textvariable=variable, font=("Cascadia Mono", 9), fg=WORKBENCH["text"], bg=WORKBENCH["header"], anchor="w").pack(fill="x", pady=(1, 0))
            self.reference_values[name] = variable
        self.support_banner = tk.Label(
            reference_row, text="Input support has not been evaluated", font=("Segoe UI Semibold", 8),
            fg=WORKBENCH["muted"], bg=WORKBENCH["chrome_dark"], anchor="w", padx=7, pady=4,
        )
        self.support_banner.pack(fill="x")

        viewport_bar = tk.Frame(body, bg=WORKBENCH["panel"], height=29)
        viewport_bar.pack(fill="x")
        viewport_bar.pack_propagate(False)
        tk.Label(viewport_bar, text="Drying-shrinkage response", font=("Segoe UI Semibold", 8), fg=WORKBENCH["text"], bg=WORKBENCH["panel"]).pack(side="left", padx=8)
        tk.Label(viewport_bar, text="X: drying age (d)    Y: shrinkage (µε)", font=("Segoe UI", 7), fg=WORKBENCH["muted"], bg=WORKBENCH["panel"]).pack(side="right", padx=8)
        tk.Frame(body, bg=WORKBENCH["border"], height=1).pack(fill="x")
        self.figure = Figure(figsize=(7.2, 5.2), dpi=100, facecolor=WORKBENCH["panel"])
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=body)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=4)

        table_bar = tk.Frame(self.table_tab, bg=WORKBENCH["header"], height=34)
        table_bar.pack(fill="x")
        table_bar.pack_propagate(False)
        tk.Label(table_bar, text="Evaluated values at standard and requested ages", font=("Segoe UI", 8), fg=WORKBENCH["text"], bg=WORKBENCH["header"]).pack(side="left", padx=8)
        result_table_frame = tk.Frame(self.table_tab, bg=WORKBENCH["panel"])
        result_table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.result_table = ttk.Treeview(
            result_table_frame, columns=("age", "pea", "sd", "b3", "gl", "aci"),
            show="headings", style="Workbench.Treeview",
        )
        for key, label, width in (
            ("age", "Drying age (d)", 105), ("pea", "PEA-PGNN (µε)", 125),
            ("sd", "Seed SD (µε)", 100), ("b3", "Model B3 (µε)", 110),
            ("gl", "GL2000 (µε)", 110), ("aci", "ACI 209 (µε)", 110),
        ):
            self.result_table.heading(key, text=label)
            self.result_table.column(key, width=width, minwidth=width, anchor="e", stretch=False)
        table_scroll = ttk.Scrollbar(result_table_frame, orient="vertical", command=self.result_table.yview)
        table_xscroll = ttk.Scrollbar(result_table_frame, orient="horizontal", command=self.result_table.xview)
        self.result_table.configure(yscrollcommand=table_scroll.set, xscrollcommand=table_xscroll.set)
        self.result_table.grid(row=0, column=0, sticky="nsew")
        table_scroll.grid(row=0, column=1, sticky="ns")
        table_xscroll.grid(row=1, column=0, sticky="ew")
        result_table_frame.grid_rowconfigure(0, weight=1)
        result_table_frame.grid_columnconfigure(0, weight=1)

        self.diagnostics_text = tk.Text(
            self.diagnostics_tab, wrap="word", font=("Cascadia Mono", 9),
            fg=WORKBENCH["text"], bg=WORKBENCH["console"], relief="flat", padx=12, pady=10,
        )
        self.diagnostics_text.insert("end", "No diagnostics available. Run the model with F5.\n")
        self.diagnostics_text.configure(state="disabled")
        self.diagnostics_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_interpretation(self, parent):
        inspector = ttk.Notebook(parent, style="Workbench.TNotebook")
        inspector.pack(fill="both", expand=True)
        result_tab = tk.Frame(inspector, bg=WORKBENCH["panel"])
        allocation_tab = tk.Frame(inspector, bg=WORKBENCH["panel"])
        inspector.add(result_tab, text="Result Inspector")
        inspector.add(allocation_tab, text="Allocation")

        inspector_frame = tk.Frame(result_tab, bg=WORKBENCH["panel"])
        inspector_frame.pack(fill="both", expand=True, padx=4, pady=(4, 0))
        self.inspector_table = ttk.Treeview(
            inspector_frame, columns=("property", "value"), show="headings",
            style="Workbench.Treeview", selectmode="browse",
        )
        self.inspector_table.heading("property", text="Property")
        self.inspector_table.heading("value", text="Value")
        self.inspector_table.column("property", width=155, anchor="w", stretch=True)
        self.inspector_table.column("value", width=170, anchor="w", stretch=True)
        self.inspector_table.tag_configure("section", background=WORKBENCH["header"], foreground=WORKBENCH["accent_dark"], font=("Segoe UI Semibold", 8))
        inspector_scroll = ttk.Scrollbar(inspector_frame, orient="vertical", command=self.inspector_table.yview)
        self.inspector_table.configure(yscrollcommand=inspector_scroll.set)
        self.inspector_table.pack(side="left", fill="both", expand=True)
        inspector_scroll.pack(side="right", fill="y")
        self.inspector_table.insert("", "end", values=("Status", "Waiting for calculation"))
        tk.Label(
            result_tab,
            text="Reference equations are comparison quantities; seed variation is not a prediction interval.",
            wraplength=320, justify="left", font=("Segoe UI", 7), fg=WORKBENCH["muted"],
            bg=WORKBENCH["header"], padx=7, pady=6,
        ).pack(fill="x", padx=4, pady=4)

        self.weight_widgets = []
        for name in ("B3-type", "ACI-type", "GL-type", "Bounded logarithmic"):
            row = tk.Frame(allocation_tab, bg=WORKBENCH["panel"])
            row.pack(fill="x", padx=10, pady=8)
            top = tk.Frame(row, bg=WORKBENCH["panel"])
            top.pack(fill="x")
            tk.Label(top, text=name, font=("Segoe UI", 8), fg=WORKBENCH["text"], bg=WORKBENCH["panel"]).pack(side="left")
            variable = tk.StringVar(value="-")
            tk.Label(top, textvariable=variable, font=("Cascadia Mono", 8), fg=WORKBENCH["accent"], bg=WORKBENCH["panel"]).pack(side="right")
            progress = ttk.Progressbar(row, maximum=1.0, value=0.0, style="Weight.Horizontal.TProgressbar")
            progress.pack(fill="x", pady=(2, 0))
            self.weight_widgets.append((progress, variable))
        tk.Label(
            allocation_tab,
            text="Internal model weights; not physical mechanism probabilities.",
            wraplength=300, justify="left", font=("Segoe UI", 7),
            fg=WORKBENCH["muted"], bg=WORKBENCH["header"], padx=8, pady=6,
        ).pack(fill="x", side="bottom", padx=8, pady=8)

    def _build_console(self, parent):
        frame = tk.Frame(parent, bg=WORKBENCH["panel"])
        frame.pack(fill="both", expand=True)
        self.console = tk.Text(
            frame, height=4, wrap="none", font=("Cascadia Mono", 8),
            fg=WORKBENCH["text"], bg=WORKBENCH["console"], relief="flat", padx=7, pady=5,
        )
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.console.yview)
        self.console.configure(yscrollcommand=scrollbar.set)
        self.console.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.console.insert("end", "[READY] PEA-PGNN V1.0.0 initialized. Enter an input condition and press F5.\n")
        self.console.configure(state="disabled")

    def _log(self, level, text):
        if not hasattr(self, "console"):
            return
        self.console.configure(state="normal")
        self.console.insert("end", "[{}] {}\n".format(level.upper(), text))
        self.console.see("end")
        self.console.configure(state="disabled")

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=WORKBENCH["chrome_dark"], height=24)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer, textvariable=self.status_text, font=("Segoe UI", 8), fg=WORKBENCH["text"], bg=WORKBENCH["chrome_dark"], anchor="w").pack(side="left", fill="x", expand=True, padx=7)
        for text_value in (
            "Device: {}".format(str(self.predictor.device).upper()),
            "Ensemble: {}".format(len(self.predictor.models)),
            "V1.0.0",
        ):
            tk.Frame(footer, bg=WORKBENCH["border"], width=1).pack(side="right", fill="y")
            tk.Label(footer, text=text_value, font=("Segoe UI", 8), fg=WORKBENCH["text"], bg=WORKBENCH["chrome_dark"], padx=10).pack(side="right")

    def _update_derived_vs(self):
        if not hasattr(self, "derived_vs"):
            return
        try:
            self.derived_vs.set("{:.2f} mm".format(float(self.fields["h0"].get()) / 2.0))
        except ValueError:
            self.derived_vs.set("-")

    def condition(self):
        condition = {}
        numeric = ("cement", "water", "aggregate", "wb", "fc28", "Ec28", "t0", "RH", "T", "h0", "query_age")
        for key in numeric:
            condition[key] = float(self.fields[key].get())
        for key in ("cement_type_code", "agg_type_code", "curing_type_code"):
            condition[key] = int(self.fields[key].get())
        condition["geometry"] = self.fields["geometry"].get()
        return condition

    def run_prediction(self):
        try:
            self.status_text.set("Running prediction...")
            self.root.update_idletasks()
            condition = self.condition()
            result = self.predictor.predict_curve(condition)
            query = float(condition["query_age"])
            query_index = int(np.where(np.isclose(result["ages"], query, atol=1e-6))[0][0])
            value = float(result["prediction"][query_index])
            variation = float(result["optimization_sd"][query_index])
            self.last_condition = condition
            self.last_result = result
            self.result_value.set("eps_sh({:g} d) = {:.1f} {}".format(query, value, MICROSTRAIN))
            self._draw_result_value(query, value)
            self.result_age.set("Optimization-seed SD: {:.1f} {} | not a prediction interval".format(variation, MICROSTRAIN))
            for name, values in result["references"].items():
                if name in self.reference_values:
                    self.reference_values[name].set("{:.1f} {}".format(float(values[query_index]), MICROSTRAIN))
            self._update_support_banner(result["support"])
            self._update_chart(result, query)
            self._update_interpretation(result, query_index, condition)
            self._update_weights(result["weights"][query_index])
            self._update_result_table(result)
            self._update_diagnostics(result, query_index, condition)
            self.status_text.set("Completed | t = {:g} d | eps_sh = {:.1f} {} | {}".format(query, value, MICROSTRAIN, result["support"]["label"]))
            self._log("OK", "Prediction completed at t = {:g} d: {:.1f} {} ({})".format(query, value, MICROSTRAIN, result["support"]["label"]))
        except Exception as exc:
            self.status_text.set("Prediction failed")
            self._log("ERROR", str(exc))
            messagebox.showerror("Prediction error", str(exc), parent=self.root)

    def _update_result_table(self, result):
        for item in self.result_table.get_children():
            self.result_table.delete(item)
        ages = np.asarray(result["ages"], dtype=float)
        requested = [7.0, 14.0, 28.0, 56.0, 90.0, 180.0, 365.0, float(self.last_condition["query_age"])]
        selected = sorted({float(age) for age in requested if np.any(np.isclose(ages, age, atol=1.0e-6))})
        for age in selected:
            index = int(np.where(np.isclose(ages, age, atol=1.0e-6))[0][0])
            refs = result["references"]
            self.result_table.insert("", "end", values=(
                "{:g}".format(age),
                "{:.1f}".format(float(result["prediction"][index])),
                "{:.1f}".format(float(result["optimization_sd"][index])),
                "{:.1f}".format(float(refs["Model B3"][index])),
                "{:.1f}".format(float(refs["GL2000"][index])),
                "{:.1f}".format(float(refs["ACI 209"][index])),
            ))

    def _update_diagnostics(self, result, query_index, condition):
        weights = result["weights"][query_index]
        names = ("B3-type", "ACI-type", "GL-type", "Bounded logarithmic")
        lines = [
            "PEA-PGNN MODEL DIAGNOSTICS",
            "=" * 58,
            "Condition age             {:>12g} d".format(float(condition["query_age"])),
            "Prediction                {:>12.2f} {}".format(float(result["prediction"][query_index]), MICROSTRAIN),
            "Optimization-seed SD      {:>12.2f} {}  [not PI]".format(float(result["optimization_sd"][query_index]), MICROSTRAIN),
            "Support classification    {:>12}".format(result["support"]["label"]),
            "",
            "BOUNDED OPERATIONAL QUANTITIES",
            "eps_anchor                {:>12.2f} {}".format(float(result["eps_anchor"][query_index]), MICROSTRAIN),
            "eps_inf                   {:>12.2f} {}".format(float(result["eps_inf"][query_index]), MICROSTRAIN),
            "tau_anchor                {:>12.2f} d".format(float(result["tau_anchor"][query_index])),
            "tau                       {:>12.2f} d".format(float(result["tau"][query_index])),
            "",
            "CANDIDATE-LAW ALLOCATION",
        ]
        lines.extend("{:<26}{:>11.2%}".format(name, float(weight)) for name, weight in zip(names, weights))
        lines.extend((
            "",
            "NOTE",
            "Weights are internal model allocations, not physical-mechanism probabilities.",
            "The reported seed variation is not a calibrated prediction interval.",
        ))
        self.diagnostics_text.configure(state="normal")
        self.diagnostics_text.delete("1.0", "end")
        self.diagnostics_text.insert("end", "\n".join(lines))
        self.diagnostics_text.configure(state="disabled")

    def _update_support_banner(self, support):
        styles = {
            "within": (COLORS["green_light"], COLORS["green"], "OK  "),
            "boundary": (COLORS["orange_light"], COLORS["orange"], "!  "),
            "outside": (COLORS["red_light"], COLORS["red"], "WARNING  "),
        }
        background, foreground, prefix = styles[support["level"]]
        details = support["outside_variables"] or support["boundary_variables"]
        suffix = " | " + ", ".join(details) if details else ""
        self.support_banner.configure(text=prefix + support["label"] + suffix, bg=background, fg=foreground)

    def _draw_result_value(self, query_age=None, value=None):
        self.result_axes.clear()
        self.result_axes.axis("off")
        if query_age is None or value is None:
            text_value = "Run a prediction"
        else:
            text_value = r"$\varepsilon_{\mathrm{sh}}(" + "{:g}".format(query_age) + r"\,\mathrm{d}) = " + "{:.1f}".format(value) + r"\,\mu\varepsilon$"
        self.result_axes.set_facecolor(WORKBENCH["header"])
        self.result_figure.set_facecolor(WORKBENCH["header"])
        self.result_axes.text(0.03, 0.52, text_value, ha="left", va="center", fontsize=15, color=WORKBENCH["accent_dark"])
        self.result_figure.tight_layout(pad=0.1)
        self.result_canvas.draw_idle()

    def _update_chart(self, result, query_age):
        self.axes.clear()
        ages = result["ages"]
        self.axes.set_facecolor("#FFFFFF")
        self.axes.plot(ages, result["prediction"], color=WORKBENCH["accent"], linewidth=2.2, label="PEA-PGNN")
        definitions = {item["name"]: item for item in self.predictor.formula_definitions()}
        style_map = {"-": "-", "--": "--", "-.": "-.", ":": ":", "long-dash": (0, (4, 2))}
        for name, values in result["references"].items():
            definition = definitions.get(name, {})
            self.axes.plot(
                ages,
                values,
                color=definition.get("color", "#7A5C3E"),
                linestyle=style_map.get(definition.get("line_style", "--"), "--"),
                linewidth=1.35,
                label=name,
            )
        index = int(np.where(np.isclose(ages, query_age, atol=1e-6))[0][0])
        self.axes.axvline(query_age, color="#98A2B3", linestyle=":", linewidth=1.0)
        self.axes.scatter([query_age], [result["prediction"][index]], s=55, color=COLORS["blue"], zorder=5)
        self.axes.annotate(
            "{:.1f} {}".format(float(result["prediction"][index]), MICROSTRAIN),
            (query_age, result["prediction"][index]), xytext=(8, 8), textcoords="offset points",
            fontsize=8, color=COLORS["blue_dark"], fontweight="bold",
        )
        self.axes.set_xlabel(r"Drying age, $t$ (d)", fontsize=9)
        self.axes.set_ylabel(r"Drying-shrinkage magnitude, $\varepsilon_{\mathrm{sh}}$ ($\mu\varepsilon$)", fontsize=9)
        self.axes.grid(True, linestyle=":", linewidth=0.55, color="#BEC4CB", alpha=0.75)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        legend = self.axes.legend(loc="lower right", fontsize=7, frameon=True, ncol=2)
        legend.get_frame().set_edgecolor(WORKBENCH["border"])
        legend.get_frame().set_linewidth(0.7)
        self.axes.set_xlim(left=0)
        self.axes.set_ylim(bottom=0)
        self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    def _update_interpretation(self, result, query_index, condition):
        prediction = float(result["prediction"][query_index])
        support = result["support"]
        eps_anchor = float(result["eps_anchor"][query_index])
        eps_inf = float(result["eps_inf"][query_index])
        tau_anchor = float(result["tau_anchor"][query_index])
        tau = float(result["tau"][query_index])
        weights = result["weights"][query_index]
        dominant_names = ["B3-type", "ACI-type", "GL-type", "bounded logarithmic"]
        dominant = dominant_names[int(np.argmax(weights))]
        references = {
            name: float(values[query_index])
            for name, values in result["references"].items()
            if name in {"Model B3", "GL2000", "ACI 209"}
        }
        comparison = "; ".join(
            "{}: {:+.1f} {}".format(name, prediction - value, MICROSTRAIN) for name, value in references.items()
        )
        for item in self.inspector_table.get_children():
            self.inspector_table.delete(item)
        rows = (
            ("RESULT", "", "section"),
            ("Drying age", "{:g} d".format(condition["query_age"]), ""),
            ("PEA-PGNN", "{:.1f} {}".format(prediction, MICROSTRAIN), ""),
            ("Seed variation", "{:.1f} {} [not PI]".format(float(result["optimization_sd"][query_index]), MICROSTRAIN), ""),
            ("Support", support["label"], ""),
            ("REFERENCE DIFFERENCE", "PEA-PGNN minus equation", "section"),
            ("Model B3", "{:+.1f} {}".format(prediction - references["Model B3"], MICROSTRAIN), ""),
            ("GL2000", "{:+.1f} {}".format(prediction - references["GL2000"], MICROSTRAIN), ""),
            ("ACI 209", "{:+.1f} {}".format(prediction - references["ACI 209"], MICROSTRAIN), ""),
            ("BOUNDED QUANTITIES", "Operational values", "section"),
            ("eps_anchor", "{:.1f} {}".format(eps_anchor, MICROSTRAIN), ""),
            ("eps_inf", "{:.1f} {}".format(eps_inf, MICROSTRAIN), ""),
            ("tau_anchor", "{:.1f} d".format(tau_anchor), ""),
            ("tau", "{:.1f} d".format(tau), ""),
            ("ALLOCATION", "", "section"),
            ("Dominant candidate", dominant, ""),
            ("Dominant weight", "{:.1%}".format(float(np.max(weights))), ""),
        )
        for property_name, value, tag in rows:
            self.inspector_table.insert("", "end", values=(property_name, value), tags=((tag,) if tag else ()))

    def _update_weights(self, weights):
        for value, (progress, variable) in zip(weights, self.weight_widgets):
            progress.configure(value=float(value))
            variable.set("{:.1%}".format(float(value)))

    def open_formula_library(self):
        FormulaLibraryDialog(self.root, self.predictor, self._formulas_changed, condition_provider=self.condition)

    def _formulas_changed(self):
        if self.last_condition is not None:
            self.run_prediction()

    def _reset_chart(self):
        self.axes.clear()
        self.axes.text(
            0.5, 0.5, "Enter a condition and run prediction", transform=self.axes.transAxes,
            ha="center", va="center", color="#808A95", fontsize=10,
        )
        self.axes.set_xlabel(r"Drying age, $t$ (d)", fontsize=9)
        self.axes.set_ylabel(r"Drying-shrinkage magnitude, $\varepsilon_{\mathrm{sh}}$ ($\mu\varepsilon$)", fontsize=9)
        self.axes.grid(True, linestyle=":", linewidth=0.55, color="#C6CBD1", alpha=0.7)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.canvas.draw_idle()

    def reset(self):
        defaults = {
            "cement": 371, "water": 186, "aggregate": 1859, "wb": 0.48, "fc28": 37,
            "Ec28": 25958, "cement_type_code": 2, "agg_type_code": 1, "t0": 7,
            "RH": 50, "T": 23, "curing_type_code": 1, "h0": 45.5,
            "geometry": "Prism", "query_age": 365,
        }
        for key, value in defaults.items():
            self.fields[key].set(str(value))
        self.result_value.set("Run a prediction")
        self._draw_result_value()
        self.result_age.set("Point prediction | nominal interval not reported")
        for variable in self.reference_values.values():
            variable.set("-")
        self.support_banner.configure(text="Input support has not been evaluated", bg=WORKBENCH["chrome_dark"], fg=WORKBENCH["muted"])
        self.last_result = None
        self.last_condition = None
        self.status_text.set("Ready")
        if hasattr(self, "result_table"):
            for item in self.result_table.get_children():
                self.result_table.delete(item)
        if hasattr(self, "diagnostics_text"):
            self.diagnostics_text.configure(state="normal")
            self.diagnostics_text.delete("1.0", "end")
            self.diagnostics_text.insert("end", "No diagnostics available. Run the model with F5.\n")
            self.diagnostics_text.configure(state="disabled")
        self._log("READY", "Input condition reset to defaults.")
        self._reset_chart()

    def export_curve(self):
        if self.last_result is None:
            messagebox.showwarning("No result", "Run a prediction before exporting.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Export prediction curve", defaultextension=".csv",
            initialfile="pea_pgnn_prediction_curve.csv", filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        result = self.last_result
        frame = pd.DataFrame(
            {
                "drying_age_d": result["ages"],
                "PEA_PGNN_microstrain": result["prediction"],
                "optimization_seed_sd_microstrain_not_PI": result["optimization_sd"],
                **{name.replace(" ", "_") + "_microstrain": values for name, values in result["references"].items()},
            }
        )
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("Export complete", "Saved:\n" + path, parent=self.root)

    def export_report(self):
        if self.last_result is None or self.last_condition is None:
            messagebox.showwarning(
                "No result",
                "Run a prediction before generating a report. The report always uses the current calculated result.",
                parent=self.root,
            )
            return
        ReportExportDialog(self)

    def batch_prediction(self):
        path = filedialog.askopenfilename(parent=self.root, title="Load batch input", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            frame = pd.read_csv(path)
            required = {"cement", "water", "aggregate", "wb", "fc28", "Ec28", "cement_type_code", "agg_type_code", "t0", "RH", "T", "curing_type_code", "h0", "geometry", "query_age"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError("Missing batch columns: {}".format(missing))
            output = self.predictor.predict_batch(frame)
            save = filedialog.asksaveasfilename(
                parent=self.root, title="Save batch results", defaultextension=".csv",
                initialfile="pea_pgnn_batch_results.csv", filetypes=[("CSV", "*.csv")],
            )
            if save:
                output.to_csv(save, index=False, encoding="utf-8-sig")
                messagebox.showinfo("Batch complete", "Saved {} rows:\n{}".format(len(output), save), parent=self.root)
        except Exception as exc:
            messagebox.showerror("Batch error", str(exc), parent=self.root)


class ReportExportDialog:
    """Small property dialog for a deterministic engineering PDF report."""

    def __init__(self, app):
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Generate calculation report - PEA-PGNN V1.0.0")
        self.window.geometry("680x585")
        self.window.minsize(630, 535)
        self.window.configure(bg=WORKBENCH["workspace"])
        self.window.transient(app.root)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.variables = {
            "title": tk.StringVar(value="Drying-Shrinkage Calculation Report"),
            "project": tk.StringVar(value=""),
            "report_id": tk.StringVar(value=make_report_id(app.last_condition, app.last_result)),
            "prepared_by": tk.StringVar(value=""),
        }
        self.report_mode = tk.StringVar(value="standard")
        self._build()
        self.window.grab_set()
        self.window.focus_set()

    def _build(self):
        toolbar = tk.Frame(self.window, bg=WORKBENCH["chrome"], height=43)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)
        tk.Label(
            toolbar, text="PDF CALCULATION REPORT", font=("Segoe UI Semibold", 10),
            fg=WORKBENCH["accent_dark"], bg=WORKBENCH["chrome"],
        ).pack(side="left", padx=11)
        tk.Label(
            toolbar, text="Current calculated result", font=("Segoe UI", 8),
            fg=WORKBENCH["muted"], bg=WORKBENCH["chrome"],
        ).pack(side="right", padx=11)
        tk.Frame(self.window, bg=WORKBENCH["border"], height=1).pack(fill="x")

        body = tk.Frame(self.window, bg=WORKBENCH["workspace"])
        body.pack(fill="both", expand=True, padx=10, pady=10)
        properties = WorkbenchPanel(body, "Document Properties")
        properties.pack(fill="both", expand=True)
        form = properties.body
        fields = (
            ("Report title", "title"),
            ("Project / specimen", "project"),
            ("Report ID", "report_id"),
            ("Prepared by", "prepared_by"),
        )
        for row_index, (label, key) in enumerate(fields):
            tk.Label(
                form, text=label, font=("Segoe UI", 9), fg=WORKBENCH["text"],
                bg=WORKBENCH["panel"], anchor="w",
            ).grid(row=row_index, column=0, sticky="w", padx=(10, 8), pady=(9 if row_index == 0 else 4, 4))
            ttk.Entry(form, textvariable=self.variables[key], style="PEA.TEntry").grid(
                row=row_index, column=1, sticky="ew", padx=(0, 10), pady=(9 if row_index == 0 else 4, 4)
            )
        form.grid_columnconfigure(1, weight=1)
        mode_row = len(fields)
        tk.Label(
            form, text="Report contents", font=("Segoe UI", 9), fg=WORKBENCH["text"],
            bg=WORKBENCH["panel"], anchor="nw",
        ).grid(row=mode_row, column=0, sticky="nw", padx=(10, 8), pady=4)
        mode_box = tk.Frame(form, bg=WORKBENCH["panel"])
        mode_box.grid(row=mode_row, column=1, sticky="ew", padx=(0, 10), pady=2)
        tk.Radiobutton(
            mode_box, text="Standard engineering report", variable=self.report_mode, value="standard",
            font=("Segoe UI", 9), fg=WORKBENCH["text"], bg=WORKBENCH["panel"],
            activebackground=WORKBENCH["panel"], selectcolor="white", anchor="w",
        ).pack(fill="x")
        tk.Radiobutton(
            mode_box, text="Complete technical report (adds model audit appendix)",
            variable=self.report_mode, value="technical", font=("Segoe UI", 9),
            fg=WORKBENCH["text"], bg=WORKBENCH["panel"],
            activebackground=WORKBENCH["panel"], selectcolor="white", anchor="w",
        ).pack(fill="x")

        notes_row = mode_row + 1
        tk.Label(
            form, text="Project notes", font=("Segoe UI", 9), fg=WORKBENCH["text"],
            bg=WORKBENCH["panel"], anchor="nw",
        ).grid(row=notes_row, column=0, sticky="nw", padx=(10, 8), pady=4)
        self.notes = tk.Text(
            form, height=5, wrap="word", undo=True, font=("Segoe UI", 9),
            fg=WORKBENCH["text"], bg="white", relief="solid", bd=1,
            highlightthickness=0,
        )
        self.notes.grid(row=notes_row, column=1, sticky="nsew", padx=(0, 10), pady=4)
        form.grid_rowconfigure(notes_row, weight=1)
        tk.Label(
            form,
            text="Standard is intended for routine engineering use. Complete technical adds internal coefficients, "
                 "input-domain diagnostics, model build records and disabled user-formula records.",
            font=("Segoe UI", 8), fg=WORKBENCH["muted"], bg=WORKBENCH["panel"],
            justify="left", wraplength=470, anchor="w",
        ).grid(row=notes_row + 1, column=0, columnspan=2, sticky="ew", padx=10, pady=(4, 9))

        warning = tk.Frame(body, bg=WORKBENCH["panel"], highlightbackground=WORKBENCH["border"], highlightthickness=1)
        warning.pack(fill="x", pady=(8, 0))
        tk.Label(
            warning,
            text="The report records the result currently shown. Generating it does not rerun or modify the calculation model.",
            font=("Segoe UI", 8), fg=WORKBENCH["muted"], bg=WORKBENCH["panel"], anchor="w",
        ).pack(fill="x", padx=8, pady=6)

        actions = tk.Frame(self.window, bg=WORKBENCH["chrome"], height=51)
        actions.pack(fill="x", side="bottom")
        actions.pack_propagate(False)
        tk.Frame(actions, bg=WORKBENCH["border"], height=1).pack(fill="x")
        self.app._tool_button(actions, "Cancel", self.close).pack(side="right", padx=(3, 10), pady=9)
        self.app._tool_button(actions, "Generate PDF...", self.generate, primary=True).pack(side="right", padx=3, pady=9)

    def close(self):
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self.window.destroy()

    def generate(self):
        metadata = {key: variable.get().strip() for key, variable in self.variables.items()}
        metadata["notes"] = self.notes.get("1.0", "end-1c").strip()
        report_id = metadata["report_id"] or make_report_id(self.app.last_condition, self.app.last_result)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", report_id).strip("._") or "PEA_PGNN_Report"
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save calculation report",
            defaultextension=".pdf",
            initialfile=safe_id + ".pdf",
            filetypes=[("PDF report", "*.pdf")],
        )
        if not path:
            return
        try:
            self.app.status_text.set("Generating PDF calculation report...")
            self.app.root.update_idletasks()
            record = build_pdf_report(
                path,
                self.app.predictor,
                self.app.last_condition,
                self.app.last_result,
                metadata=metadata,
                report_mode=self.report_mode.get(),
            )
            self.app.status_text.set("Report generated | {}".format(record.report_id))
            self.app._log("OK", "PDF report generated: {} | SHA-256 {}".format(record.path, record.sha256[:12]))
            self.close()
            if messagebox.askyesno(
                "Report generated",
                "The PDF calculation report was saved successfully.\n\n{}\n\nOpen it now?".format(record.path),
                parent=self.app.root,
            ):
                try:
                    os.startfile(str(record.path))
                except OSError as exc:
                    messagebox.showwarning("Unable to open PDF", "The report was saved, but Windows could not open it:\n{}".format(exc), parent=self.app.root)
        except Exception as exc:
            self.app.status_text.set("Report generation failed")
            self.app._log("ERROR", "PDF report: {}".format(exc))
            messagebox.showerror("Report error", str(exc), parent=self.window)


class FormulaLibraryDialog:
    """Formula catalogue, notation viewer and safe package manager."""

    def __init__(self, parent, predictor, on_change, condition_provider=None):
        self.parent = parent
        self.predictor = predictor
        self.on_change = on_change
        self.condition_provider = condition_provider
        self.definitions = []
        self.window = tk.Toplevel(parent)
        self.window.title("Formula library - PEA-PGNN V1.0.0")
        self.window.geometry("1050x680")
        self.window.minsize(900, 600)
        self.window.configure(bg=COLORS["background"])
        self.window.transient(parent)
        self._build()
        self.refresh()

    def _build(self):
        toolbar = tk.Frame(self.window, bg=COLORS["background"])
        toolbar.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(
            toolbar,
            text="Embedded formula library",
            font=("Segoe UI Semibold", 17),
            fg=COLORS["blue_dark"],
            bg=COLORS["background"],
        ).pack(side="left")
        self._tool_button(toolbar, "Import file", self.import_package).pack(side="right", padx=(6, 0))
        self._tool_button(toolbar, "New formula", self.new_formula).pack(side="right", padx=(6, 0))
        self._tool_button(toolbar, "Reload", self.refresh).pack(side="right")

        note = (
            "Built-in formulas marked 'model prior' are part of the frozen 39-variable model. "
            "They are read-only and cannot be edited, disabled or removed. User formulas are kept in a separate recoverable data space."
        )
        tk.Label(
            self.window, text=note, wraplength=1000, justify="left", anchor="w",
            font=FONT, fg=COLORS["text"], bg="#EEF3F8", padx=12, pady=9,
        ).pack(fill="x", padx=16, pady=(0, 10))

        split = tk.PanedWindow(self.window, orient="horizontal", sashwidth=5, bg=COLORS["border"], bd=0)
        split.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        left = tk.Frame(split, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        right = tk.Frame(split, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        split.add(left, width=330, minsize=280)
        split.add(right, minsize=500)

        tk.Label(left, text="Available formulas", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=12, pady=(11, 8))
        self.listbox = tk.Listbox(
            left, exportselection=False, font=FONT, relief="flat", bd=0,
            selectbackground="#DCE8F6", selectforeground=COLORS["text"], activestyle="none",
        )
        self.listbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.listbox.bind("<<ListboxSelect>>", self._show_selected)
        self.error_label = tk.Label(left, text="", wraplength=300, justify="left", font=FONT_SMALL, fg=COLORS["red"], bg=COLORS["card"])
        self.error_label.pack(fill="x", padx=12, pady=(0, 8))

        self.title_var = tk.StringVar()
        self.meta_var = tk.StringVar()
        tk.Label(right, textvariable=self.title_var, font=("Segoe UI Semibold", 15), fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=18, pady=(15, 2))
        tk.Label(right, textvariable=self.meta_var, font=FONT_SMALL, fg=COLORS["muted"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=18)
        self.math_figure = Figure(figsize=(6.4, 1.5), dpi=100, facecolor="white")
        self.math_axes = self.math_figure.add_subplot(111)
        self.math_canvas = FigureCanvasTkAgg(self.math_figure, master=right)
        self.math_canvas.get_tk_widget().pack(fill="x", padx=12, pady=(8, 4))
        self.detail = tk.Text(
            right, wrap="word", font=FONT, fg=COLORS["text"], bg=COLORS["card"],
            relief="flat", padx=8, pady=5, height=14,
        )
        self.detail.tag_configure("heading", font=("Segoe UI Semibold", 10), foreground=COLORS["blue_dark"], spacing1=8, spacing3=2)
        self.detail.tag_configure("code", font=("Cascadia Mono", 9), foreground="#3E4C59", background="#F5F7FA", lmargin1=8, lmargin2=8, spacing1=3, spacing3=5)
        self.detail.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        actions = tk.Frame(right, bg=COLORS["card"])
        actions.pack(fill="x", padx=18, pady=(0, 14))
        self.edit_button = self._tool_button(actions, "Edit", self.edit_selected)
        self.edit_button.pack(side="left")
        self.copy_button = self._tool_button(actions, "Copy as custom", self.copy_selected)
        self.copy_button.pack(side="left", padx=(8, 0))
        self.toggle_button = self._tool_button(actions, "Disable", self.toggle_selected)
        self.toggle_button.pack(side="left", padx=(8, 0))
        self.remove_button = self._tool_button(actions, "Remove", self.remove_selected)
        self.remove_button.pack(side="left", padx=(8, 0))
        self._tool_button(actions, "Restore backup", self.restore_archived).pack(side="right", padx=(8, 0))
        self._tool_button(actions, "Open user formula space", self.open_folder).pack(side="right")

    def _tool_button(self, parent, text, command):
        return tk.Button(
            parent, text=text, command=command, font=("Segoe UI", 9),
            fg=COLORS["blue_dark"], bg="white", activebackground=COLORS["blue_light"],
            relief="solid", bd=1, cursor="hand2", padx=10, pady=5,
        )

    def refresh(self, select_id=None):
        self.definitions = self.predictor.reload_formulas()
        self.listbox.delete(0, "end")
        selected_index = 0
        for index, item in enumerate(self.definitions):
            state = "enabled" if item["enabled"] else "disabled"
            marker = "built-in" if item["locked"] else "custom"
            self.listbox.insert("end", "{}  [{}; {}]".format(item["name"], marker, state))
            if item["id"] == select_id:
                selected_index = index
        errors = self.predictor.formulas.errors
        self.error_label.configure(text="{} invalid user package(s) were safely quarantined.".format(len(errors)) if errors else "")
        if self.definitions:
            self.listbox.selection_set(selected_index)
            self.listbox.activate(selected_index)
            self._show_selected()

    def _selected(self):
        selection = self.listbox.curselection()
        return self.definitions[int(selection[0])] if selection else None

    def _show_selected(self, event=None):
        item = self._selected()
        if item is None:
            return
        self.title_var.set(item["name"])
        prior = "model prior" if item.get("model_prior") else "comparison only"
        protection = "protected; read-only" if item["locked"] else "user formula"
        self.meta_var.set("{} | {} | {} | {} | {}".format(
            item["source"], prior, item["output_unit"], protection,
            "enabled" if item["enabled"] else "disabled",
        ))
        self.math_axes.clear()
        self.math_axes.axis("off")
        latex = item.get("latex", "").strip()
        if latex:
            try:
                self.math_axes.text(0.5, 0.5, "$" + latex.strip("$") + "$", ha="center", va="center", fontsize=14)
            except Exception:
                self.math_axes.text(0.02, 0.5, latex, ha="left", va="center", fontsize=11)
        else:
            self.math_axes.text(0.02, 0.5, "No typeset notation supplied", ha="left", va="center", fontsize=10, color=COLORS["muted"])
        self.math_figure.tight_layout(pad=0.5)
        self.math_canvas.draw_idle()

        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("end", "Purpose\n", "heading")
        self.detail.insert("end", item.get("description", "No description supplied.") + "\n")
        self.detail.insert("end", "Role in this software\n", "heading")
        self.detail.insert("end", item["role"] + "\n")
        if item["locked"]:
            self.detail.insert("end", "Protection\n", "heading")
            self.detail.insert("end", "This native definition is compiled into the application. Edit, disable and archive actions are locked. Use 'Copy as custom' to experiment without changing the original.\n")
        else:
            self.detail.insert("end", "Storage and recovery\n", "heading")
            self.detail.insert("end", "This user formula is stored outside the application. Archive is recoverable, and edits keep revision copies.\n")
        self.detail.insert("end", "Expression\n", "heading")
        self.detail.insert("end", item.get("expression", ""), "code")
        constants = item.get("constants", {})
        if constants:
            self.detail.insert("end", "\nConstants\n", "heading")
            self.detail.insert("end", "\n".join("{} = {}".format(key, value) for key, value in constants.items()), "code")
        self.detail.configure(state="disabled")
        custom = not item["locked"]
        self.edit_button.configure(state=("normal" if custom else "disabled"))
        self.copy_button.configure(state="normal")
        self.toggle_button.configure(state=("normal" if custom else "disabled"), text=("Disable" if item["enabled"] else "Enable"))
        self.remove_button.configure(state=("normal" if custom else "disabled"), text="Archive")

    def import_package(self):
        path = filedialog.askopenfilename(parent=self.window, title="Import formula package", filetypes=[("PEA formula", "*.peaf"), ("JSON", "*.json")])
        if not path:
            return
        try:
            item = self.predictor.import_formula(path)
            self.refresh(item["id"])
            self.on_change()
        except Exception as exc:
            messagebox.showerror("Formula import error", str(exc), parent=self.window)

    def new_formula(self):
        FormulaEditorDialog(
            self.window, self.predictor, None, self._editor_saved,
            condition_provider=self.condition_provider,
        )

    def edit_selected(self):
        item = self._selected()
        if item is None or item["locked"]:
            return
        FormulaEditorDialog(
            self.window, self.predictor, item, self._editor_saved,
            condition_provider=self.condition_provider,
        )

    def copy_selected(self):
        item = self._selected()
        if item is None:
            return
        copied = dict(item)
        copied["id"] = re.sub(r"[^a-z0-9_]+", "_", item["id"].lower()).strip("_") + "_copy"
        copied["name"] = item["name"] + " - custom copy"
        copied["locked"] = False
        copied["model_prior"] = False
        copied["role"] = "Comparison curve only (model retraining required for use as a prior)"
        if item["locked"]:
            copied["expression"] = FormulaEditorDialog.STARTER_EXPRESSION
            copied["latex"] = FormulaEditorDialog.STARTER_LATEX
            copied["constants"] = {"eps_u": 1000.0, "tau": 55.0}
            copied["description"] = "Editable comparison curve copied from {}. The built-in source remains unchanged.".format(item["name"])
        FormulaEditorDialog(
            self.window, self.predictor, copied, self._editor_saved,
            copy_mode=True, condition_provider=self.condition_provider,
        )

    def _editor_saved(self, formula_id):
        self.refresh(formula_id)
        self.on_change()

    def export_template(self):
        path = filedialog.asksaveasfilename(parent=self.window, title="Export formula template", defaultextension=".peaf", initialfile="my_formula.peaf", filetypes=[("PEA formula", "*.peaf")])
        if not path:
            return
        try:
            FormulaRegistry.write_template(path)
            messagebox.showinfo("Template exported", "Saved:\n" + path, parent=self.window)
        except Exception as exc:
            messagebox.showerror("Export error", str(exc), parent=self.window)

    def toggle_selected(self):
        item = self._selected()
        if item is None or item["locked"]:
            return
        try:
            self.predictor.set_formula_enabled(item["id"], not item["enabled"])
            self.refresh(item["id"])
            self.on_change()
        except Exception as exc:
            messagebox.showerror("Formula error", str(exc), parent=self.window)

    def remove_selected(self):
        item = self._selected()
        if item is None or item["locked"]:
            return
        if not messagebox.askyesno(
            "Archive formula",
            "Move '{}' out of the active formula library?\n\nIt will not be permanently deleted and can be restored later.".format(item["name"]),
            parent=self.window,
        ):
            return
        try:
            destination = self.predictor.remove_formula(item["id"])
            self.refresh()
            self.on_change()
            messagebox.showinfo(
                "Formula archived",
                "The formula was removed from active comparisons but remains recoverable:\n{}".format(destination),
                parent=self.window,
            )
        except Exception as exc:
            messagebox.showerror("Formula error", str(exc), parent=self.window)

    def restore_archived(self):
        FormulaBackupDialog(self.window, self.predictor, self._backup_restored)

    def _backup_restored(self, formula_id):
        self.refresh(formula_id)
        self.on_change()

    def open_folder(self):
        try:
            os.startfile(str(self.predictor.formulas.directory))
        except Exception as exc:
            messagebox.showerror("Open folder error", str(exc), parent=self.window)


class FormulaBackupDialog:
    """One-click backup recovery for users who should not handle package paths."""

    def __init__(self, parent, predictor, on_restored):
        self.predictor = predictor
        self.on_restored = on_restored
        self.items = []
        self.window = tk.Toplevel(parent)
        self.window.title("Restore user formula backup - PEA-PGNN V1.0.0")
        self.window.geometry("760x470")
        self.window.minsize(680, 400)
        self.window.configure(bg=COLORS["background"])
        self.window.transient(parent)
        self._build()
        self.refresh()

    def _build(self):
        tk.Label(
            self.window, text="Restore user formula backup", font=("Segoe UI Semibold", 17),
            fg=COLORS["blue_dark"], bg=COLORS["background"], anchor="w",
        ).pack(fill="x", padx=18, pady=(15, 4))
        tk.Label(
            self.window,
            text="Select an available backup and press Restore. Native formulas are stored inside the application and never appear here.",
            wraplength=710, justify="left", font=FONT, fg=COLORS["text"], bg="#EEF3F8", padx=10, pady=8,
        ).pack(fill="x", padx=18, pady=(0, 10))

        frame = tk.Frame(self.window, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        columns = ("name", "source", "saved", "status")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for key, label, width in (
            ("name", "Formula", 240), ("source", "Backup type", 100),
            ("saved", "Saved", 155), ("status", "Recovery status", 180),
        ):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, anchor=("w" if key in {"name", "status"} else "center"), stretch=True)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=8)
        self.table.bind("<<TreeviewSelect>>", self._selection_changed)
        self.table.bind("<Double-1>", lambda event: self.restore_selected())

        bottom = tk.Frame(self.window, bg=COLORS["background"])
        bottom.pack(fill="x", padx=18, pady=(0, 13))
        self.status_var = tk.StringVar(value="Loading recoverable backups...")
        tk.Label(bottom, textvariable=self.status_var, font=FONT_SMALL, fg=COLORS["muted"], bg=COLORS["background"], anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(
            bottom, text="Close", command=self.window.destroy, font=("Segoe UI Semibold", 9),
            fg=COLORS["blue_dark"], bg="white", activebackground=COLORS["blue_light"],
            relief="solid", bd=1, padx=16, pady=6,
        ).pack(side="right")
        self.restore_button = tk.Button(
            bottom, text="Restore selected", command=self.restore_selected, font=("Segoe UI Semibold", 9),
            fg="white", bg=COLORS["blue"], activebackground=COLORS["blue_dark"], activeforeground="white",
            relief="solid", bd=1, padx=16, pady=6, state="disabled",
        )
        self.restore_button.pack(side="right", padx=(0, 8))

    def refresh(self):
        self.items = self.predictor.formulas.backups()
        for row in self.table.get_children():
            self.table.delete(row)
        available = 0
        for index, item in enumerate(self.items):
            status = "Ready to restore" if item["restorable"] else "Active formula already exists"
            if item["restorable"]:
                available += 1
            self.table.insert("", "end", iid=str(index), values=(item["name"], item["kind"], item["saved_at"], status))
        if self.items:
            first = next((index for index, item in enumerate(self.items) if item["restorable"]), 0)
            self.table.selection_set(str(first))
            self.table.focus(str(first))
            self._selection_changed()
            self.status_var.set("{} backup(s); {} currently available for recovery.".format(len(self.items), available))
        else:
            self.status_var.set("No user-formula backups have been created yet.")

    def _selected(self):
        selection = self.table.selection()
        return self.items[int(selection[0])] if selection else None

    def _selection_changed(self, event=None):
        item = self._selected()
        self.restore_button.configure(state=("normal" if item and item["restorable"] else "disabled"))

    def restore_selected(self):
        item = self._selected()
        if item is None or not item["restorable"]:
            return
        try:
            restored = self.predictor.restore_formula(item["path"])
            self.on_restored(restored["id"])
            messagebox.showinfo(
                "Formula restored",
                "'{}' is active again. Its backup history remains protected.".format(restored["name"]),
                parent=self.window,
            )
            self.window.destroy()
        except Exception as exc:
            messagebox.showerror("Restore formula error", str(exc), parent=self.window)


class TrialCalculationDialog:
    """Engineering trial calculation for one unsaved custom formula."""

    CONDITION_FIELDS = (
        ("RH", "Relative humidity, RH", "%"),
        ("T", "Temperature, T", "deg C"),
        ("h0", "Theoretical thickness, h₀", "mm"),
        ("t0", "Curing age, t₀", "d"),
        ("wb", "Water-binder ratio, w/b", "-"),
        ("fc28", "Compressive strength, fᶜ₂₈", "MPa"),
        ("Ec28", "Elastic modulus, Eᶜ₂₈", "MPa"),
        ("cement", "Cement", "kg/m3"),
        ("water", "Water", "kg/m3"),
        ("aggregate", "Aggregate", "kg/m3"),
    )
    SENSITIVITY_LABELS = {
        "RH": "Relative humidity, RH (%)",
        "T": "Temperature, T (deg C)",
        "h0": "Theoretical thickness, h0 (mm)",
        "t0": "Curing age, t0 (d)",
        "wb": "Water-binder ratio, w/b",
        "fc28": "Compressive strength, fc28 (MPa)",
        "Ec28": "Elastic modulus, Ec28 (MPa)",
        "cement": "Cement (kg/m3)",
        "water": "Water (kg/m3)",
        "aggregate": "Aggregate (kg/m3)",
    }

    def __init__(self, parent, predictor, document, condition_provider=None):
        self.predictor = predictor
        self.document = document
        self.condition_provider = condition_provider
        self.condition_vars = {}
        self.last_single_frame = None
        self.last_sensitivity_frame = None
        self.base_condition = dict(DEFAULT_TRIAL_CONDITION)
        self.window = tk.Toplevel(parent)
        self.window.title("Formula trial calculation - PEA-PGNN V1.0.0")
        self.window.geometry("1180x800")
        self.window.minsize(1040, 700)
        self.window.configure(bg=COLORS["background"])
        self.window.transient(parent)
        self._build()
        self.load_main_condition(run=False)
        self.window.after(120, self.run_single)

    def _build(self):
        heading = tk.Frame(self.window, bg=COLORS["background"])
        heading.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(
            heading, text="Trial calculation", font=("Segoe UI Semibold", 18),
            fg=COLORS["blue_dark"], bg=COLORS["background"],
        ).pack(side="left")
        tk.Label(
            heading, text=self.document.get("name", "Custom formula"), font=("Segoe UI Semibold", 11),
            fg=COLORS["text"], bg=COLORS["background"],
        ).pack(side="left", padx=(16, 0), pady=(6, 0))
        tk.Label(
            heading, text="Numerical screening; engineering validation is still required.", font=FONT_SMALL,
            fg=COLORS["muted"], bg=COLORS["background"],
        ).pack(side="right", pady=(8, 0))

        self.notebook = ttk.Notebook(self.window)
        notebook = self.notebook
        notebook.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        single = tk.Frame(notebook, bg=COLORS["background"])
        sensitivity = tk.Frame(notebook, bg=COLORS["background"])
        notebook.add(single, text="  Single-condition trial  ")
        notebook.add(sensitivity, text="  Sensitivity trial  ")
        self._build_single_tab(single)
        self._build_sensitivity_tab(sensitivity)

        footer = tk.Frame(self.window, bg=COLORS["background"])
        footer.pack(fill="x", padx=18, pady=(0, 12), side="bottom", before=notebook)
        tk.Button(
            footer, text="Close", command=self.window.destroy, font=("Segoe UI Semibold", 9),
            fg=COLORS["blue_dark"], bg="white", activebackground=COLORS["blue_light"],
            relief="solid", bd=1, padx=18, pady=6,
        ).pack(side="right")

    def _build_single_tab(self, parent):
        condition_card = tk.Frame(parent, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        condition_card.pack(fill="x", pady=(8, 8))
        title_row = tk.Frame(condition_card, bg=COLORS["card"])
        title_row.pack(fill="x", padx=14, pady=(10, 5))
        tk.Label(title_row, text="Trial condition", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"]).pack(side="left")
        tk.Button(
            title_row, text="Load main-window inputs", command=self.load_main_condition,
            font=("Segoe UI", 9), fg=COLORS["blue_dark"], bg="white",
            activebackground=COLORS["blue_light"], relief="solid", bd=1, padx=10, pady=3,
        ).pack(side="right")

        fields = tk.Frame(condition_card, bg=COLORS["card"])
        fields.pack(fill="x", padx=14, pady=(0, 8))
        for index, (key, label, unit) in enumerate(self.CONDITION_FIELDS):
            column = index % 5
            row = (index // 5) * 2
            tk.Label(fields, text=label, font=FONT_SMALL, fg=COLORS["text"], bg=COLORS["card"], anchor="w").grid(
                row=row, column=column, sticky="ew", padx=(0, 12), pady=(0, 2)
            )
            cell = tk.Frame(fields, bg=COLORS["card"])
            cell.grid(row=row + 1, column=column, sticky="ew", padx=(0, 12), pady=(0, 7))
            variable = tk.StringVar()
            self.condition_vars[key] = variable
            ttk.Entry(cell, textvariable=variable, width=12, justify="right").pack(side="left", fill="x", expand=True)
            tk.Label(cell, text=unit, font=FONT_SMALL, fg=COLORS["muted"], bg=COLORS["card"], width=7, anchor="w").pack(side="left", padx=(4, 0))
            fields.grid_columnconfigure(column, weight=1)

        settings = tk.Frame(condition_card, bg="#F6F8FA")
        settings.pack(fill="x", padx=14, pady=(0, 11))
        tk.Label(settings, text="Geometry", font=FONT_SMALL, fg=COLORS["text"], bg="#F6F8FA").pack(side="left", padx=(8, 5), pady=8)
        self.geometry_var = tk.StringVar(value="Prism")
        ttk.Combobox(settings, textvariable=self.geometry_var, values=["Prism", "Cylinder"], state="readonly", width=10).pack(side="left")
        tk.Label(settings, text="Report ages", font=FONT_SMALL, fg=COLORS["text"], bg="#F6F8FA").pack(side="left", padx=(18, 5))
        self.ages_var = tk.StringVar(value="7, 14, 28, 56, 90, 180, 365")
        ttk.Entry(settings, textvariable=self.ages_var, width=31).pack(side="left")
        tk.Label(settings, text="Curve to", font=FONT_SMALL, fg=COLORS["text"], bg="#F6F8FA").pack(side="left", padx=(18, 5))
        self.maximum_age_var = tk.StringVar(value="400")
        ttk.Entry(settings, textvariable=self.maximum_age_var, width=8, justify="right").pack(side="left")
        tk.Label(settings, text="d", font=FONT_SMALL, fg=COLORS["muted"], bg="#F6F8FA").pack(side="left", padx=(4, 0))
        tk.Button(
            settings, text="Calculate", command=self.run_single, font=("Segoe UI Semibold", 9),
            fg="white", bg=COLORS["blue"], activebackground=COLORS["blue_dark"],
            activeforeground="white", relief="solid", bd=1, padx=18, pady=5,
        ).pack(side="right", padx=7, pady=5)

        result_split = tk.PanedWindow(parent, orient="horizontal", sashwidth=5, bg=COLORS["border"], bd=0)
        result_split.pack(fill="both", expand=True, pady=(0, 8))
        left = tk.Frame(result_split, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        right = tk.Frame(result_split, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        result_split.add(left, width=525, minsize=490)
        result_split.add(right, minsize=570)

        tk.Label(left, text="Key-age results", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=12, pady=(10, 5))
        table_frame = tk.Frame(left, bg=COLORS["card"])
        table_frame.pack(fill="x", padx=10, pady=(0, 5))
        columns = ("age", "formula", "pea", "b3", "gl", "aci")
        self.single_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        headings = (("age", "Age (d)", 62), ("formula", "Custom", 82), ("pea", "PEA-PGNN", 88), ("b3", "B3", 70), ("gl", "GL2000", 76), ("aci", "ACI 209", 76))
        for key, label, width in headings:
            self.single_table.heading(key, text=label)
            self.single_table.column(key, width=width, minwidth=55, anchor="e", stretch=True)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.single_table.xview)
        self.single_table.configure(xscrollcommand=xscroll.set)
        self.single_table.pack(fill="x")
        xscroll.pack(fill="x")

        tk.Label(left, text="Formula health check", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=12, pady=(8, 3))
        self.health_text = tk.Text(left, height=10, wrap="word", font=FONT_SMALL, relief="flat", bg="#F7F9FB", padx=8, pady=5)
        self.health_text.tag_configure("pass", foreground=COLORS["green"], font=("Segoe UI Semibold", 8))
        self.health_text.tag_configure("warn", foreground=COLORS["orange"], font=("Segoe UI Semibold", 8))
        self.health_text.tag_configure("fail", foreground=COLORS["red"], font=("Segoe UI Semibold", 8))
        self.health_text.tag_configure("detail", foreground=COLORS["text"], spacing3=2)
        self.health_text.pack(fill="both", expand=True, padx=10, pady=(0, 7))

        tk.Label(right, text="Formula and model comparison", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=12, pady=(10, 3))
        action = tk.Frame(right, bg=COLORS["card"])
        action.pack(fill="x", padx=12, pady=(2, 8), side="bottom")
        self.single_status_var = tk.StringVar(value="Preparing trial calculation...")
        tk.Label(action, textvariable=self.single_status_var, font=FONT_SMALL, fg=COLORS["muted"], bg=COLORS["card"], anchor="w").pack(side="left", fill="x", expand=True)
        tk.Button(
            action, text="Export CSV", command=lambda: self.export_frame(self.last_single_frame, "formula_trial_calculation.csv"),
            font=("Segoe UI", 9), fg=COLORS["blue_dark"], bg="white", activebackground=COLORS["blue_light"],
            relief="solid", bd=1, padx=10, pady=4,
        ).pack(side="right")
        self.single_figure = Figure(figsize=(6.4, 4.4), dpi=100, facecolor="white")
        self.single_axes = self.single_figure.add_subplot(111)
        self.single_canvas = FigureCanvasTkAgg(self.single_figure, master=right)
        self.single_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 3))

    def _build_sensitivity_tab(self, parent):
        controls = tk.Frame(parent, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        controls.pack(fill="x", pady=(8, 8))
        tk.Label(
            controls, text="Change one quantity at a time; all other entries come from the single-condition tab.",
            font=FONT, fg=COLORS["text"], bg=COLORS["card"],
        ).pack(anchor="w", padx=14, pady=(10, 7))
        row = tk.Frame(controls, bg=COLORS["card"])
        row.pack(fill="x", padx=14, pady=(0, 11))
        sensitivity_options = dict(self.SENSITIVITY_LABELS)
        for name in self.document.get("constants", {}):
            sensitivity_options["constant:" + name] = "Formula parameter: {}".format(name)
        self.sensitivity_options = sensitivity_options
        self.sensitivity_label_to_key = {label: key for key, label in sensitivity_options.items()}
        default_key = "RH" if "RH" in self.document.get("expression", "") else next(iter(sensitivity_options))
        self.sensitivity_variable_var = tk.StringVar(value=sensitivity_options[default_key])
        tk.Label(row, text="Quantity", font=FONT_SMALL, fg=COLORS["text"], bg=COLORS["card"]).pack(side="left")
        ttk.Combobox(
            row, textvariable=self.sensitivity_variable_var, values=list(sensitivity_options.values()),
            state="readonly", width=34,
        ).pack(side="left", padx=(6, 18))
        tk.Label(row, text="Trial values", font=FONT_SMALL, fg=COLORS["text"], bg=COLORS["card"]).pack(side="left")
        self.sensitivity_values_var = tk.StringVar(value="40, 50, 60, 70, 80")
        ttk.Entry(row, textvariable=self.sensitivity_values_var, width=28).pack(side="left", padx=(6, 18))
        tk.Label(row, text="Curve to", font=FONT_SMALL, fg=COLORS["text"], bg=COLORS["card"]).pack(side="left")
        self.sensitivity_max_age_var = tk.StringVar(value="400")
        ttk.Entry(row, textvariable=self.sensitivity_max_age_var, width=8, justify="right").pack(side="left", padx=(6, 3))
        tk.Label(row, text="d", font=FONT_SMALL, fg=COLORS["muted"], bg=COLORS["card"]).pack(side="left")
        tk.Button(
            row, text="Run sensitivity", command=self.run_sensitivity, font=("Segoe UI Semibold", 9),
            fg="white", bg=COLORS["blue"], activebackground=COLORS["blue_dark"], activeforeground="white",
            relief="solid", bd=1, padx=15, pady=5,
        ).pack(side="right")

        split = tk.PanedWindow(parent, orient="horizontal", sashwidth=5, bg=COLORS["border"], bd=0)
        split.pack(fill="both", expand=True, pady=(0, 8))
        left = tk.Frame(split, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        right = tk.Frame(split, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        split.add(left, width=350, minsize=300)
        split.add(right, minsize=600)
        tk.Label(left, text="Scenario summary", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=12, pady=(10, 5))
        sens_columns = ("setting", "d28", "d90", "last")
        self.sensitivity_table = ttk.Treeview(left, columns=sens_columns, show="headings", height=13)
        for key, label, width in (("setting", "Setting", 110), ("d28", "28 d", 70), ("d90", "90 d", 70), ("last", "Final", 75)):
            self.sensitivity_table.heading(key, text=label)
            self.sensitivity_table.column(key, width=width, anchor="e", stretch=True)
        self.sensitivity_table.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.sensitivity_status_var = tk.StringVar(value="Choose a quantity and run the sensitivity trial.")
        tk.Label(left, textvariable=self.sensitivity_status_var, wraplength=320, justify="left", font=FONT_SMALL, fg=COLORS["muted"], bg="#F7F9FB", padx=8, pady=7).pack(fill="x", padx=10, pady=(0, 7))
        tk.Button(
            left, text="Export CSV", command=lambda: self.export_frame(self.last_sensitivity_frame, "formula_sensitivity_trial.csv"),
            font=("Segoe UI", 9), fg=COLORS["blue_dark"], bg="white", activebackground=COLORS["blue_light"],
            relief="solid", bd=1, padx=10, pady=4,
        ).pack(anchor="e", padx=10, pady=(0, 9))

        tk.Label(right, text="Sensitivity curves", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=12, pady=(10, 3))
        self.sensitivity_figure = Figure(figsize=(7.0, 5.2), dpi=100, facecolor="white")
        self.sensitivity_axes = self.sensitivity_figure.add_subplot(111)
        self.sensitivity_canvas = FigureCanvasTkAgg(self.sensitivity_figure, master=right)
        self.sensitivity_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._reset_axes(self.sensitivity_axes, "Run sensitivity to compare formula responses")
        self.sensitivity_canvas.draw_idle()

    def load_main_condition(self, run=True):
        condition = dict(DEFAULT_TRIAL_CONDITION)
        note = "Loaded default demonstration condition."
        if self.condition_provider is not None:
            try:
                condition.update(self.condition_provider())
                note = "Loaded the current inputs from the main window."
            except Exception as exc:
                note = "Main-window inputs are incomplete; defaults were loaded ({})".format(exc)
        self.base_condition = condition
        for key, _, _ in self.CONDITION_FIELDS:
            self.condition_vars[key].set("{:g}".format(float(condition[key])))
        self.geometry_var.set(str(condition.get("geometry", "Prism")))
        if hasattr(self, "single_status_var"):
            self.single_status_var.set(note)
        if run:
            self.run_single()

    def _condition(self):
        condition = dict(self.base_condition)
        for key, _, _ in self.CONDITION_FIELDS:
            try:
                condition[key] = float(self.condition_vars[key].get())
            except ValueError as exc:
                raise ValueError("{} must be a number".format(key)) from exc
        condition["geometry"] = self.geometry_var.get()
        for key in ("cement_type_code", "agg_type_code", "curing_type_code"):
            condition[key] = int(condition.get(key, DEFAULT_TRIAL_CONDITION[key]))
        return condition

    @staticmethod
    def _reset_axes(axes, message):
        axes.clear()
        axes.text(0.5, 0.5, message, transform=axes.transAxes, ha="center", va="center", color="#98A2B3", fontsize=11)
        axes.set_xlabel(r"Drying age, $t$ (d)", fontsize=9)
        axes.set_ylabel(r"$\varepsilon_{\mathrm{sh}}$ ($\mu\varepsilon$)", fontsize=9)
        axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)

    @staticmethod
    def _reference(result, name):
        values = result.get("references", {}).get(name)
        if values is None:
            return np.full(len(result["ages"]), np.nan)
        return np.asarray(values, dtype=float)

    def run_single(self):
        try:
            condition = self._condition()
            report_ages = parse_number_list(self.ages_var.get(), maximum_count=20)
            maximum_age = max(float(self.maximum_age_var.get()), float(np.max(report_ages)))
            model_maximum = float(self.predictor.manifest.get("max_query_age_days", 4832.92))
            if maximum_age > model_maximum:
                raise ValueError("Curve age exceeds the implemented model maximum of {:.2f} d".format(model_maximum))
            curve_ages = build_trial_age_grid(maximum_age, report_ages)
            condition["query_age"] = maximum_age
            _, custom_curve = self.predictor.preview_formula(self.document, condition, curve_ages)
            result = self.predictor.predict(condition, curve_ages, include_details=False)
            references = {name: self._reference(result, name) for name in ("Model B3", "GL2000", "ACI 209")}

            zero_value = None
            try:
                _, at_zero = self.predictor.preview_formula(self.document, condition, [0.0])
                zero_value = float(at_zero[0])
            except Exception:
                zero_value = None
            health = analyse_formula_curve(curve_ages, custom_curve, zero_value=zero_value)
            self._show_health(health)

            for item in self.single_table.get_children():
                self.single_table.delete(item)
            for age in report_ages:
                index = int(np.where(np.isclose(curve_ages, age, rtol=0, atol=1.0e-8))[0][0])
                row = [
                    "{:g}".format(age), "{:.1f}".format(custom_curve[index]), "{:.1f}".format(result["prediction"][index]),
                    "{:.1f}".format(references["Model B3"][index]), "{:.1f}".format(references["GL2000"][index]),
                    "{:.1f}".format(references["ACI 209"][index]),
                ]
                self.single_table.insert("", "end", values=row)

            self.single_axes.clear()
            self.single_axes.plot(curve_ages, custom_curve, color=self.document.get("color", "#8B5E3C"), linewidth=2.5, label=self.document["name"])
            self.single_axes.plot(curve_ages, result["prediction"], color=COLORS["blue"], linewidth=2.3, label="PEA-PGNN")
            styles = {"Model B3": ("#2E7D32", "--"), "GL2000": ("#C25400", "-."), "ACI 209": ("#6B3FA0", ":")}
            for name, values in references.items():
                self.single_axes.plot(curve_ages, values, color=styles[name][0], linestyle=styles[name][1], linewidth=1.25, alpha=0.9, label=name)
            self.single_axes.set_xlabel(r"Drying age, $t$ (d)", fontsize=9)
            self.single_axes.set_ylabel(r"Drying-shrinkage magnitude, $\varepsilon_{\mathrm{sh}}$ ($\mu\varepsilon$)", fontsize=9)
            self.single_axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
            self.single_axes.spines["top"].set_visible(False)
            self.single_axes.spines["right"].set_visible(False)
            self.single_axes.legend(fontsize=7, ncol=2, frameon=True)
            self.single_figure.tight_layout(pad=1.0)
            self.single_canvas.draw_idle()

            self.last_single_frame = pd.DataFrame({
                "drying_age_d": curve_ages,
                self.document["name"] + "_microstrain": custom_curve,
                "PEA_PGNN_microstrain": result["prediction"],
                **{name.replace(" ", "_") + "_microstrain": values for name, values in references.items()},
            })
            if health["failure_count"]:
                summary = "Trial completed with {} failed check(s) and {} warning(s).".format(health["failure_count"], health["warning_count"])
            elif health["warning_count"]:
                summary = "Trial completed: {} review warning(s); no numerical failure.".format(health["warning_count"])
            else:
                summary = "Trial completed: all sampled numerical checks passed."
            self.single_status_var.set(summary)
        except Exception as exc:
            self.last_single_frame = None
            self.single_status_var.set("Trial could not be completed: " + str(exc))
            self._reset_axes(self.single_axes, "Correct the highlighted trial inputs and calculate again")
            self.single_figure.tight_layout(pad=1.0)
            self.single_canvas.draw_idle()

    def _show_health(self, health):
        self.health_text.configure(state="normal")
        self.health_text.delete("1.0", "end")
        labels = {"pass": "PASS", "warn": "REVIEW", "fail": "FAIL"}
        for item in health["checks"]:
            self.health_text.insert("end", "{}  {} - ".format(labels[item["level"]], item["title"]), item["level"])
            self.health_text.insert("end", item["detail"] + "\n", "detail")
        self.health_text.configure(state="disabled")

    @staticmethod
    def _sample_value(ages, values, target):
        target = min(float(target), float(ages[-1]))
        return float(np.interp(target, ages, values))

    def run_sensitivity(self):
        try:
            key = self.sensitivity_label_to_key[self.sensitivity_variable_var.get()]
            values = parse_number_list(self.sensitivity_values_var.get(), positive=False, maximum_count=10)
            maximum_age = float(self.sensitivity_max_age_var.get())
            model_maximum = float(self.predictor.manifest.get("max_query_age_days", 4832.92))
            if maximum_age <= 0 or maximum_age > model_maximum:
                raise ValueError("Curve age must lie between 0 and {:.2f} d".format(model_maximum))
            ages = build_trial_age_grid(maximum_age, [28.0, 90.0, maximum_age])
            base_condition = self._condition()
            base_condition["query_age"] = maximum_age
            curves = {}
            for value in values:
                condition = dict(base_condition)
                document = dict(self.document)
                document["constants"] = dict(self.document.get("constants", {}))
                if key.startswith("constant:"):
                    document["constants"][key.split(":", 1)[1]] = float(value)
                else:
                    condition[key] = float(value)
                self.predictor.validate_condition(condition)
                _, curve = self.predictor.preview_formula(document, condition, ages)
                curves[float(value)] = curve

            for item in self.sensitivity_table.get_children():
                self.sensitivity_table.delete(item)
            for value, curve in curves.items():
                self.sensitivity_table.insert("", "end", values=(
                    "{:g}".format(value),
                    "{:.1f}".format(self._sample_value(ages, curve, 28)),
                    "{:.1f}".format(self._sample_value(ages, curve, 90)),
                    "{:.1f}".format(curve[-1]),
                ))

            self.sensitivity_axes.clear()
            colours = matplotlib.colormaps["viridis"](np.linspace(0.12, 0.88, len(curves)))
            for colour, (value, curve) in zip(colours, curves.items()):
                self.sensitivity_axes.plot(ages, curve, color=colour, linewidth=2.0, label="{} = {:g}".format(key.replace("constant:", ""), value))
            self.sensitivity_axes.set_xlabel(r"Drying age, $t$ (d)", fontsize=9)
            self.sensitivity_axes.set_ylabel(r"$\varepsilon_{\mathrm{sh}}$ ($\mu\varepsilon$)", fontsize=9)
            self.sensitivity_axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
            self.sensitivity_axes.spines["top"].set_visible(False)
            self.sensitivity_axes.spines["right"].set_visible(False)
            self.sensitivity_axes.legend(fontsize=8, frameon=True)
            self.sensitivity_figure.tight_layout(pad=1.0)
            self.sensitivity_canvas.draw_idle()
            self.last_sensitivity_frame = pd.DataFrame({
                "drying_age_d": ages,
                **{"{}_{:g}_microstrain".format(key.replace("constant:", ""), value): curve for value, curve in curves.items()},
            })
            span = float(max(curve[-1] for curve in curves.values()) - min(curve[-1] for curve in curves.values()))
            self.sensitivity_status_var.set(
                "{} scenario(s) completed. Final-age result span: {:.1f} {}. This is a one-at-a-time screen, not a global sensitivity analysis.".format(len(curves), span, MICROSTRAIN)
            )
        except Exception as exc:
            self.last_sensitivity_frame = None
            self.sensitivity_status_var.set("Sensitivity trial could not be completed: " + str(exc))
            self._reset_axes(self.sensitivity_axes, "Correct the settings and run sensitivity again")
            self.sensitivity_figure.tight_layout(pad=1.0)
            self.sensitivity_canvas.draw_idle()

    def export_frame(self, frame, initialfile):
        if frame is None:
            messagebox.showwarning("No trial result", "Run this trial before exporting.", parent=self.window)
            return
        path = filedialog.asksaveasfilename(
            parent=self.window, title="Export trial results", defaultextension=".csv",
            initialfile=initialfile, filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("Export complete", "Saved:\n" + path, parent=self.window)


class FormulaEditorDialog:
    """Guided formula editor for users who do not write code."""

    STARTER_EXPRESSION = "eps_u*(1-(RH/100)**humidity_power)*(t/(tau+t))**time_power"
    STARTER_LATEX = r"\varepsilon_{\mathrm{sh}}(t)=\varepsilon_{\mathrm{u}}[1-(RH/100)^{n}]\left(\frac{t}{\tau+t}\right)^{m}"
    TEMPLATES = {
        "Exponential growth": {
            "expression": "eps_u*(1-(RH/100)**humidity_power)*(1-exp(-t/tau))",
            "latex": r"\varepsilon_{\mathrm{sh}}(t)=\varepsilon_{\mathrm{u}}[1-(RH/100)^{n}][1-e^{-t/\tau}]",
            "constants": {"eps_u": 1000.0, "humidity_power": 3.0, "tau": 55.0},
        },
        "Hyperbolic growth": {
            "expression": "eps_u*(1-(RH/100)**humidity_power)*(t/(tau+t))**time_power",
            "latex": STARTER_LATEX,
            "constants": {"eps_u": 1000.0, "humidity_power": 3.0, "tau": 55.0, "time_power": 0.5},
        },
        "Square-root size effect": {
            "expression": "eps_u*(1-(RH/100)**humidity_power)*sqrt(t/(t+size_factor*VtoS**2))",
            "latex": r"\varepsilon_{\mathrm{sh}}(t)=\varepsilon_{\mathrm{u}}[1-(RH/100)^{n}]\sqrt{\frac{t}{t+k(V/S)^2}}",
            "constants": {"eps_u": 1000.0, "humidity_power": 3.0, "size_factor": 0.15},
        },
        "Custom expression": {
            "expression": STARTER_EXPRESSION,
            "latex": STARTER_LATEX,
            "constants": {"eps_u": 1000.0, "humidity_power": 3.0, "tau": 55.0, "time_power": 0.5},
        },
    }

    VARIABLE_HELP = (
        "Available quantities: t = drying age; t0 = curing age; RH = relative humidity (%); "
        "T = temperature; h0 = theoretical thickness; VtoS = volume/surface ratio; "
        "wb = water-binder ratio; fc28 = compressive strength; cement, water, aggregate, Ec28."
    )

    def __init__(self, parent, predictor, definition, on_saved, copy_mode=False, condition_provider=None):
        self.predictor = predictor
        self.on_saved = on_saved
        self.condition_provider = condition_provider
        self.original_id = None if copy_mode or definition is None or definition.get("locked") else definition.get("id")
        self.window = tk.Toplevel(parent)
        self.window.title("Formula editor - PEA-PGNN V1.0.0")
        self.window.geometry("1100x760")
        self.window.minsize(960, 680)
        self.window.configure(bg=COLORS["background"])
        self.window.transient(parent)
        self._build()
        self._load(definition)

    def _build(self):
        top = tk.Frame(self.window, bg=COLORS["background"])
        top.pack(fill="x", padx=18, pady=(15, 9))
        tk.Label(top, text="Formula editor", font=("Segoe UI Semibold", 18), fg=COLORS["blue_dark"], bg=COLORS["background"]).pack(side="left")
        tk.Label(top, text="Create a comparison curve without editing files", font=FONT, fg=COLORS["muted"], bg=COLORS["background"]).pack(side="left", padx=(14, 0), pady=(7, 0))

        main = tk.PanedWindow(self.window, orient="horizontal", sashwidth=5, bg=COLORS["border"], bd=0)
        main.pack(fill="both", expand=True, padx=18, pady=(0, 10))
        left = tk.Frame(main, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        right = tk.Frame(main, bg=COLORS["card"], highlightbackground=COLORS["border"], highlightthickness=1)
        main.add(left, width=535, minsize=470)
        main.add(right, minsize=420)

        left_canvas = tk.Canvas(left, bg=COLORS["card"], highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left, orient="vertical", command=left_canvas.yview)
        left_form = tk.Frame(left_canvas, bg=COLORS["card"])
        left_window = left_canvas.create_window((0, 0), window=left_form, anchor="nw")
        left_form.bind("<Configure>", lambda event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<Configure>", lambda event: left_canvas.itemconfigure(left_window, width=event.width))
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        left = left_form

        self.name_var = tk.StringVar()
        self.id_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.template_var = tk.StringVar(value="Hyperbolic growth")
        self.color_var = tk.StringVar(value="#8B5E3C")
        self.style_var = tk.StringVar(value="--")
        self.mode_var = tk.StringVar(value="guided")
        self._updating_id = False
        self.status_var = tk.StringVar(value="Choose a template, adjust parameters and press Test formula.")
        self.convert_status_var = tk.StringVar(value="Paste a formula above, then press Generate calculation.")
        self.mathtype_status_var = tk.StringVar(value="Visual editing: send the formula to MathType, edit it, then read it back.")
        self.text_input_visible = False
        self._id_manually_set = self.original_id is not None

        self._entry_row(left, "Formula name", self.name_var, "A clear name shown in the chart legend")
        self._entry_row(left, "Description", self.description_var, "Optional engineering note")

        row = tk.Frame(left, bg=COLORS["card"])
        row.pack(fill="x", padx=16, pady=(7, 3))
        tk.Label(row, text="Starting form", font=("Segoe UI Semibold", 9), bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w")
        combo = ttk.Combobox(row, textvariable=self.template_var, values=list(self.TEMPLATES), state="readonly")
        combo.pack(fill="x", pady=(4, 2))
        combo.bind("<<ComboboxSelected>>", self._apply_template)

        mathtype_card = tk.Frame(left, bg="#EAF2FB", highlightbackground="#B8CCE4", highlightthickness=1)
        mathtype_card.pack(fill="x", padx=16, pady=(12, 2))
        tk.Label(mathtype_card, text="Edit visually with MathType", font=("Segoe UI Semibold", 10), fg=COLORS["blue_dark"], bg="#EAF2FB").pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(mathtype_card, text="No LaTeX editing is required. Use MathType's fraction, root, superscript and symbol palettes.", wraplength=485, justify="left", font=FONT_SMALL, fg=COLORS["muted"], bg="#EAF2FB").pack(fill="x", padx=10)
        mt_buttons = tk.Frame(mathtype_card, bg="#EAF2FB")
        mt_buttons.pack(fill="x", padx=10, pady=(8, 5))
        self._button(mt_buttons, "1  Open in MathType", self.open_in_mathtype, primary=True).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._button(mt_buttons, "2  Use edited formula", self.read_from_mathtype).pack(side="left", fill="x", expand=True, padx=(4, 0))
        tk.Label(mathtype_card, textvariable=self.mathtype_status_var, wraplength=485, justify="left", anchor="w", font=FONT_SMALL, fg=COLORS["muted"], bg="#EAF2FB").pack(fill="x", padx=10, pady=(0, 8))

        self.input_card = tk.Frame(left, bg="#F4F7FA", highlightbackground=COLORS["border"], highlightthickness=1)
        input_card = self.input_card
        input_head = tk.Frame(input_card, bg="#F4F7FA")
        input_head.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(input_head, text="Or enter a published formula", font=("Segoe UI Semibold", 10), fg=COLORS["blue_dark"], bg="#F4F7FA").pack(side="left")
        tk.Label(input_head, text="LaTeX, Unicode or calculator notation", font=FONT_SMALL, fg=COLORS["muted"], bg="#F4F7FA").pack(side="right")
        self.source_formula_text = tk.Text(input_card, height=3, wrap="word", font=("Cambria Math", 10), bg="white", relief="solid", bd=1)
        self.source_formula_text.pack(fill="x", padx=10, pady=(0, 5))
        self.source_formula_text.insert("1.0", r"\varepsilon_{sh}(t)=\varepsilon_u[1-(RH/100)^n](t/(\tau+t))^m")
        convert_row = tk.Frame(input_card, bg="#F4F7FA")
        convert_row.pack(fill="x", padx=10, pady=(0, 8))
        self._button(convert_row, "Generate calculation", self.generate_from_formula, primary=True).pack(side="left")
        tk.Label(convert_row, textvariable=self.convert_status_var, wraplength=300, justify="left", anchor="w", font=FONT_SMALL, fg=COLORS["muted"], bg="#F4F7FA").pack(side="left", fill="x", expand=True, padx=(10, 0))

        self.show_text_button = tk.Button(
            left, text="Use text or LaTeX instead", command=self.toggle_text_input,
            font=("Segoe UI", 9), fg=COLORS["blue_dark"], bg=COLORS["card"],
            activebackground=COLORS["blue_light"], relief="flat", cursor="hand2",
        )
        self.show_text_button.pack(anchor="w", padx=16, pady=(8, 0))

        tk.Label(left, text="Parameters", font=("Segoe UI Semibold", 10), fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=16, pady=(12, 5))
        self.constants_frame = tk.Frame(left, bg=COLORS["card"])
        self.constants_frame.pack(fill="x", padx=16)
        self.constant_vars = {}

        mode = tk.Frame(left, bg=COLORS["card"])
        mode.pack(fill="x", padx=16, pady=(13, 5))
        tk.Label(mode, text="Editing level", font=("Segoe UI Semibold", 9), bg=COLORS["card"], fg=COLORS["text"]).pack(side="left")
        tk.Radiobutton(mode, text="Simple", variable=self.mode_var, value="guided", command=self._switch_mode, bg=COLORS["card"], activebackground=COLORS["card"]).pack(side="left", padx=(18, 6))
        tk.Radiobutton(mode, text="Advanced", variable=self.mode_var, value="advanced", command=self._switch_mode, bg=COLORS["card"], activebackground=COLORS["card"]).pack(side="left")

        self.advanced = tk.Frame(left, bg=COLORS["card"])
        self._entry_row(self.advanced, "Formula ID", self.id_var, "Internal identifier; normally no change is required")
        self.name_var.trace_add("write", self._name_changed)
        self.id_var.trace_add("write", self._id_changed)
        tk.Label(self.advanced, text="Calculation expression", font=("Segoe UI Semibold", 9), bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w")
        self.expression_text = tk.Text(self.advanced, height=4, wrap="word", font=("Cascadia Mono", 9), bg="#F7F9FB", relief="solid", bd=1)
        self.expression_text.pack(fill="x", pady=(4, 7))
        tk.Label(self.advanced, text="Typeset notation (LaTeX)", font=("Segoe UI Semibold", 9), bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w")
        self.latex_text = tk.Text(self.advanced, height=4, wrap="word", font=("Cascadia Mono", 9), bg="#F7F9FB", relief="solid", bd=1)
        self.latex_text.pack(fill="x", pady=(4, 4))
        tk.Label(self.advanced, text=self.VARIABLE_HELP, wraplength=490, justify="left", font=FONT_SMALL, fg=COLORS["muted"], bg=COLORS["card"]).pack(fill="x")

        options = tk.Frame(left, bg=COLORS["card"])
        options.pack(fill="x", padx=16, pady=(12, 12))
        tk.Label(options, text="Line colour", font=FONT_SMALL, bg=COLORS["card"], fg=COLORS["text"]).grid(row=0, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.color_var, width=12).grid(row=0, column=1, padx=(7, 18))
        tk.Label(options, text="Line style", font=FONT_SMALL, bg=COLORS["card"], fg=COLORS["text"]).grid(row=0, column=2, sticky="w")
        ttk.Combobox(options, textvariable=self.style_var, values=["-", "--", "-.", ":", "long-dash"], state="readonly", width=12).grid(row=0, column=3, padx=(7, 0))

        tk.Label(right, text="Formula preview", font=FONT_CARD, fg=COLORS["blue_dark"], bg=COLORS["card"], anchor="w").pack(fill="x", padx=16, pady=(14, 4))
        self.preview_figure = Figure(figsize=(5.2, 2.7), dpi=100, facecolor="white")
        self.preview_axes = self.preview_figure.add_subplot(111)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=right)
        self.preview_canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=4)
        self.notation_figure = Figure(figsize=(5.2, 1.2), dpi=100, facecolor="white")
        self.notation_axes = self.notation_figure.add_subplot(111)
        self.notation_canvas = FigureCanvasTkAgg(self.notation_figure, master=right)
        self.notation_canvas.get_tk_widget().pack(fill="x", padx=12, pady=(2, 8))
        tk.Label(right, textvariable=self.status_var, wraplength=470, justify="left", anchor="w", font=FONT, fg=COLORS["muted"], bg="#F5F7FA", padx=10, pady=8).pack(fill="x", padx=12, pady=(0, 12))

        buttons = tk.Frame(self.window, bg=COLORS["background"])
        buttons.pack(fill="x", padx=18, pady=(0, 14))
        self._button(buttons, "Cancel", self.window.destroy).pack(side="right")
        self._button(buttons, "Save formula", self.save, primary=True).pack(side="right", padx=(0, 8))
        self._button(buttons, "Trial calculation...", self.open_trial_calculation).pack(side="right", padx=(0, 8))
        self._button(buttons, "Quick check", self.test_formula).pack(side="right", padx=(0, 8))

    def _entry_row(self, parent, label, variable, hint):
        row = tk.Frame(parent, bg=COLORS["card"])
        row.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(row, text=label, font=("Segoe UI Semibold", 9), bg=COLORS["card"], fg=COLORS["text"]).pack(anchor="w")
        ttk.Entry(row, textvariable=variable).pack(fill="x", pady=(4, 1))
        tk.Label(row, text=hint, font=FONT_SMALL, bg=COLORS["card"], fg=COLORS["muted"], anchor="w").pack(fill="x")

    def _button(self, parent, text, command, primary=False):
        return tk.Button(
            parent, text=text, command=command, font=("Segoe UI Semibold", 9),
            fg=("white" if primary else COLORS["blue_dark"]), bg=(COLORS["blue"] if primary else "white"),
            activebackground=(COLORS["blue_dark"] if primary else COLORS["blue_light"]),
            activeforeground=("white" if primary else COLORS["blue_dark"]), relief="solid", bd=1, padx=15, pady=7,
        )

    def _load(self, definition):
        if definition is None:
            self.name_var.set("My shrinkage formula")
            self._set_template("Hyperbolic growth")
        else:
            self.name_var.set(definition.get("name", "Custom formula"))
            self._updating_id = True
            self.id_var.set(definition.get("id", "custom_formula"))
            self._updating_id = False
            self.description_var.set(definition.get("description", ""))
            self.color_var.set(definition.get("color", "#8B5E3C"))
            self.style_var.set(definition.get("line_style", "--"))
            self.template_var.set("Custom expression")
            self._set_expression(definition.get("expression", self.STARTER_EXPRESSION), definition.get("latex", self.STARTER_LATEX))
            self._set_constants(definition.get("constants", {}))
            self.mode_var.set("advanced")
            self._switch_mode()
        self.test_formula()

    def _set_template(self, name):
        template = self.TEMPLATES[name]
        self.template_var.set(name)
        self._set_expression(template["expression"], template["latex"])
        self._set_constants(template["constants"])

    def _apply_template(self, event=None):
        self._set_template(self.template_var.get())
        self.test_formula()

    def generate_from_formula(self):
        source = self.source_formula_text.get("1.0", "end").strip()
        try:
            converted = convert_formula(source)
            self.template_var.set("Custom expression")
            self._set_expression(converted["expression"], converted["latex"])
            self._set_constants(converted["parameters"])
            self.convert_status_var.set(
                "Generated safely. Recognised {} adjustable parameter(s): {}.".format(
                    len(converted["parameters"]),
                    ", ".join(converted["parameters"]) if converted["parameters"] else "none",
                )
            )
            self.test_formula()
        except Exception as exc:
            self.convert_status_var.set("Could not convert: " + str(exc))

    def toggle_text_input(self):
        self.text_input_visible = not self.text_input_visible
        if self.text_input_visible:
            self.input_card.pack(fill="x", padx=16, pady=(6, 2), before=self.show_text_button)
            self.show_text_button.configure(text="Hide text or LaTeX input")
        else:
            self.input_card.pack_forget()
            self.show_text_button.configure(text="Use text or LaTeX instead")

    def open_in_mathtype(self):
        if find_mathtype() is None:
            messagebox.showerror("MathType not found", "Desktop MathType was not found on this computer.", parent=self.window)
            return
        source = self.latex_text.get("1.0", "end").strip()
        if self.text_input_visible:
            typed = self.source_formula_text.get("1.0", "end").strip()
            if typed:
                source = typed
        if not source:
            source = self.STARTER_LATEX
        try:
            set_clipboard_text(source)
            hwnd = launch_mathtype()
            if hwnd:
                self.window.after(350, self._paste_to_mathtype)
            else:
                self.window.after(1200, self._paste_to_mathtype)
            self.mathtype_status_var.set("MathType is opening. Edit the visual equation, then return and press 'Use edited formula'.")
        except Exception as exc:
            messagebox.showerror("MathType error", str(exc), parent=self.window)

    def _paste_to_mathtype(self):
        try:
            if not paste_current_formula():
                self.window.after(800, self._paste_to_mathtype)
        except Exception as exc:
            self.mathtype_status_var.set("MathType opened, but automatic paste failed. Press Ctrl+V in MathType: " + str(exc))

    def read_from_mathtype(self):
        try:
            copy_current_formula()
            self.window.after(250, self._finish_mathtype_read)
        except Exception as exc:
            messagebox.showerror("MathType error", str(exc), parent=self.window)

    def _finish_mathtype_read(self):
        try:
            mathml = read_mathml_clipboard()
            notation = mathml_to_formula(mathml)
            self.source_formula_text.delete("1.0", "end")
            self.source_formula_text.insert("1.0", notation)
            converted = convert_formula(notation)
            self.template_var.set("Custom expression")
            self._set_expression(converted["expression"], converted["latex"])
            self._set_constants(converted["parameters"])
            self.mathtype_status_var.set(
                "MathType formula received. Recognised parameters: {}.".format(
                    ", ".join(converted["parameters"]) if converted["parameters"] else "none"
                )
            )
            self.test_formula()
            self.window.lift()
            self.window.focus_force()
        except Exception as exc:
            messagebox.showerror("Could not read MathType formula", str(exc), parent=self.window)

    def _set_expression(self, expression, latex):
        self.expression_text.delete("1.0", "end")
        self.expression_text.insert("1.0", expression)
        self.latex_text.delete("1.0", "end")
        self.latex_text.insert("1.0", latex)

    def _set_constants(self, constants):
        for child in self.constants_frame.winfo_children():
            child.destroy()
        self.constant_vars = {}
        for row_index, (name, value) in enumerate(constants.items()):
            variable = tk.StringVar(value=str(value))
            self.constant_vars[name] = variable
            tk.Label(self.constants_frame, text=self._constant_label(name), font=FONT, bg=COLORS["card"], fg=COLORS["text"], anchor="w").grid(row=row_index, column=0, sticky="w", pady=3)
            ttk.Entry(self.constants_frame, textvariable=variable, width=17, justify="right").grid(row=row_index, column=1, sticky="ew", padx=(10, 6), pady=3)
            tk.Label(self.constants_frame, text=self._constant_unit(name), font=FONT_SMALL, bg=COLORS["card"], fg=COLORS["muted"], width=8, anchor="w").grid(row=row_index, column=2, sticky="w")
        self.constants_frame.grid_columnconfigure(1, weight=1)

    @staticmethod
    def _constant_label(name):
        return {
            "eps_u": "Ultimate magnitude, eps_u",
            "tau": "Characteristic time, tau",
            "humidity_power": "Humidity exponent, n",
            "time_power": "Time exponent, m",
            "size_factor": "Size coefficient, k",
        }.get(name, name)

    @staticmethod
    def _constant_unit(name):
        return {"eps_u": MICROSTRAIN, "tau": "d"}.get(name, "-")

    def _switch_mode(self):
        if self.mode_var.get() == "advanced":
            self.advanced.pack(fill="x", padx=16, pady=(3, 0), after=self.constants_frame)
        else:
            self.advanced.pack_forget()

    def _name_changed(self, *args):
        if self.original_id is None and not getattr(self, "_id_manually_set", False):
            value = re.sub(r"[^a-z0-9]+", "_", self.name_var.get().lower()).strip("_")
            if value and not value[0].isalpha():
                value = "formula_" + value
            self._updating_id = True
            self.id_var.set((value or "my_formula")[:48])
            self._updating_id = False

    def _id_changed(self, *args):
        if not self._updating_id and self.id_var.get():
            self._id_manually_set = True

    def _document(self):
        constants = {}
        for name, variable in self.constant_vars.items():
            try:
                constants[name] = float(variable.get())
            except ValueError:
                raise ValueError("Parameter '{}' must be a number".format(self._constant_label(name)))
        return {
            "schema_version": 1,
            "id": self.id_var.get().strip(),
            "name": self.name_var.get().strip(),
            "description": self.description_var.get().strip(),
            "expression": self.expression_text.get("1.0", "end").strip(),
            "latex": self.latex_text.get("1.0", "end").strip(),
            "constants": constants,
            "color": self.color_var.get().strip(),
            "line_style": self.style_var.get(),
            "enabled": True,
        }

    def test_formula(self):
        try:
            document = self._document()
            ages, values = self.predictor.preview_formula(document)
            self.preview_axes.clear()
            self.preview_axes.plot(ages, values, color=document["color"], linewidth=2.2, marker="o", markersize=4)
            self.preview_axes.set_xlabel(r"Drying age, $t$ (d)", fontsize=9)
            self.preview_axes.set_ylabel(r"$\varepsilon_{\mathrm{sh}}$ ($\mu\varepsilon$)", fontsize=9)
            self.preview_axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
            self.preview_axes.spines["top"].set_visible(False)
            self.preview_axes.spines["right"].set_visible(False)
            self.preview_figure.tight_layout(pad=1.0)
            self.preview_canvas.draw_idle()
            self.notation_axes.clear()
            self.notation_axes.axis("off")
            latex = document["latex"]
            self.notation_axes.text(0.5, 0.5, "$" + latex.strip("$") + "$", ha="center", va="center", fontsize=13)
            self.notation_figure.tight_layout(pad=0.4)
            self.notation_canvas.draw_idle()
            results = ", ".join("{:g} d: {:.1f} {}".format(age, value, MICROSTRAIN) for age, value in zip(ages, values))
            self.status_var.set("Formula is valid. " + results)
            return True
        except Exception as exc:
            self.status_var.set("Please check the formula: " + str(exc))
            return False

    def open_trial_calculation(self):
        if not self.test_formula():
            messagebox.showerror("Cannot run trial", self.status_var.get(), parent=self.window)
            return
        TrialCalculationDialog(
            self.window,
            self.predictor,
            self._document(),
            condition_provider=self.condition_provider,
        )

    def save(self):
        if not self.test_formula():
            messagebox.showerror("Formula not saved", self.status_var.get(), parent=self.window)
            return
        try:
            item = self.predictor.save_formula(self._document(), original_id=self.original_id)
            self.on_saved(item["id"])
            self.window.destroy()
        except Exception as exc:
            messagebox.showerror("Save formula error", str(exc), parent=self.window)


def main(
    artifact_directory,
    smoke=False,
    auto_predict=False,
    show_formulas=False,
    show_formula_editor=False,
    show_formula_trial=False,
    show_report=False,
):
    root = tk.Tk()
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = PEAPGNNApp(root, artifact_directory)
    if smoke:
        root.update_idletasks()
        app.run_prediction()
        root.update_idletasks()
        print("GUI smoke test passed: default prediction completed")
        root.destroy()
        return
    if auto_predict:
        root.after(150, app.run_prediction)
    if show_formulas:
        root.after(450, app.open_formula_library)
    if show_formula_editor:
        root.after(450, lambda: FormulaEditorDialog(
            root, app.predictor, None, lambda formula_id: None,
            condition_provider=app.condition,
        ))
    if show_formula_trial:
        trial_document = {
            "schema_version": 1,
            "id": "trial_formula",
            "name": "Trial formula",
            "description": "GUI trial-calculation diagnostic formula.",
            "expression": FormulaEditorDialog.STARTER_EXPRESSION,
            "latex": FormulaEditorDialog.STARTER_LATEX,
            "constants": {"eps_u": 1000.0, "humidity_power": 3.0, "tau": 55.0, "time_power": 0.5},
            "color": "#8B5E3C",
            "line_style": "--",
            "enabled": True,
        }
        root.after(450, lambda: TrialCalculationDialog(
            root, app.predictor, trial_document, condition_provider=app.condition,
        ))
    if show_report:
        def open_report_after_prediction():
            app.run_prediction()
            ReportExportDialog(app)
        root.after(450, open_report_after_prediction)
    root.mainloop()

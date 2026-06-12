import pathlib
import shlex
import re
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

# Dark theme color palette
DARK_BG = "#0b0f16"
PANEL_BG = "#0f1724"
ACCENT = "#5c6cff"
TEXT = "#e6eef8"
MUTED = "#9aa0b1"
SELECT_BG = "#27314a"
SELECT_FG = "#ffffff"
TOOLTIP_BG = "#222831"
TOOLTIP_FG = "#f8f8ff"

ROOT = pathlib.Path(__file__).resolve().parent
IGNORED_DIRS = {".git", "__pycache__", "venv", ".venv", "env", "envs"}


def is_ignored(path: pathlib.Path) -> bool:
    return any(part in IGNORED_DIRS for part in path.parts)


def get_description(path: pathlib.Path) -> str:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return "No description available."

    lines = []
    first_lines = text.splitlines()[:20]
    i = 0
    while i < len(first_lines):
        stripped = first_lines[i].strip()
        if stripped.startswith("#"):
            lines.append(stripped.lstrip("# ").strip())
            i += 1
            continue
        if stripped.startswith(('"""', "'''")):
            delimiter = stripped[:3]
            content = stripped[3:]
            if content.endswith(delimiter):
                return content[:-3].strip()
            lines.append(content.strip())
            i += 1
            while i < len(first_lines):
                line = first_lines[i]
                if delimiter in line:
                    lines.append(line.split(delimiter, 1)[0].strip())
                    break
                lines.append(line.strip())
                i += 1
            return " ".join([line for line in lines if line])
        break
    if lines:
        return " ".join([line for line in lines if line])
    # if no top comment, try to find inline usage hints like '# args:' or '# cmd:' in the first 80 lines
    for line in text.splitlines()[:80]:
        s = line.strip()
        if s.lower().startswith('# args:') or s.lower().startswith('# cmd:'):
            return s.lstrip('# ').strip()
    return "No description available. Add a top comment with a short summary."


def nice_title(rel_path: pathlib.Path) -> str:
    base = rel_path.stem.replace("_", " ").replace("-", " ").title()
    if rel_path.parent == pathlib.Path("."):
        return base
    category = " / ".join(part.replace("-", " ").title() for part in rel_path.parent.parts)
    return f"{category} / {base}"


def get_scripts() -> list[dict]:
    scripts = []
    for path in sorted(ROOT.rglob("*.py")):
        if path == pathlib.Path(__file__) or is_ignored(path.relative_to(ROOT)):
            continue
        rel = path.relative_to(ROOT)
        description = get_description(path)
        # detect modal entrypoints and whether file uses modal
        try:
            txt = path.read_text(errors="ignore")
        except OSError:
            txt = ""
        entrypoints = []
        lines = txt.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("@app.local_entrypoint"):
                # look ahead for the function definition
                for j in range(i + 1, min(i + 8, len(lines))):
                    m = re.match(r"\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", lines[j])
                    if m:
                        entrypoints.append(m.group(1))
                        break

        # Treat a file as a Modal app if it defines an App or uses @app/@modal decorators
        uses_modal = ("modal.App(" in txt) or ("@app." in txt) or ("@modal." in txt)

        scripts.append({
            "path": path,
            "rel": rel,
            "title": nice_title(rel),
            "description": description,
            "category": rel.parts[0] if len(rel.parts) > 1 else "Root",
            "entrypoints": entrypoints,
            "uses_modal": uses_modal,
        })
    return scripts


def win_to_wsl(path: pathlib.Path) -> str:
    drive = path.drive.rstrip(":").lower()
    windows_path = str(path).replace("\\", "/")
    if windows_path.startswith(f"{drive}:"):
        return "/mnt/" + drive + windows_path[2:]
    return windows_path


def create_command(script_path: pathlib.Path, extra_args: str, launch_mode: str = "python3", entrypoint: str | None = None) -> str:
    """
    Build a WSL command string. launch_mode: 'python3', 'modal_run', 'modal_serve', 'modal_deploy'
    If entrypoint is provided, modal run will call file::entrypoint.
    """
    root_wsl = win_to_wsl(ROOT)
    script_wsl = win_to_wsl(script_path)
    quoted_root = shlex.quote(root_wsl)
    quoted_script = shlex.quote(script_wsl)
    quoted_args = " ".join(shlex.quote(token) for token in shlex.split(extra_args)) if extra_args.strip() else ""

    if launch_mode == "python3":
        return f"cd {quoted_root} && python3 {quoted_script} {quoted_args}".strip()
    if launch_mode == "modal_run":
        if entrypoint:
            # modal run file::entrypoint -- <args>
            if quoted_args:
                return f"cd {quoted_root} && modal run {quoted_script}::{shlex.quote(entrypoint)} -- {quoted_args}"
            return f"cd {quoted_root} && modal run {quoted_script}::{shlex.quote(entrypoint)}"
        else:
            if quoted_args:
                return f"cd {quoted_root} && modal run {quoted_script} -- {quoted_args}"
            return f"cd {quoted_root} && modal run {quoted_script}"
    if launch_mode == "modal_serve":
        return f"cd {quoted_root} && modal serve {quoted_script}"
    if launch_mode == "modal_deploy":
        return f"cd {quoted_root} && modal deploy {quoted_script}"

    # fallback
    return f"cd {quoted_root} && python3 {quoted_script} {quoted_args}".strip()


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.text_func = text if callable(text) else None
        self.tipwindow = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)
        self.widget.bind("<ButtonPress>", self.hide)

    def get_text(self):
        if self.text_func:
            return self.text_func()
        return self.text

    def show(self, event=None):
        text = self.get_text()
        if self.tipwindow or not text:
            return
        x = event.x_root + 16 if event else self.widget.winfo_rootx() + 16
        y = event.y_root + 16 if event else self.widget.winfo_rooty() + 16
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=text,
            justify="left",
            background=TOOLTIP_BG,
            foreground=TOOLTIP_FG,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
        )
        label.pack(ipadx=6, ipady=4)

    def hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

    def set_text(self, text: str):
        self.text = text
        self.text_func = None


class LauncherApp(tk.Tk):
    def __init__(self, scripts):
        super().__init__()
        self.title("AlphaModal WSL Script Launcher")
        self.geometry("1100x720")
        self.minsize(980, 600)
        self.scripts = scripts
        self.selected_script = None
        # Window and default palette
        self.configure(bg=DARK_BG)
        # Apply a basic tk palette so classic tk widgets inherit dark colors
        try:
            self.option_add("*Background", DARK_BG)
            self.option_add("*Foreground", TEXT)
            self.option_add("*Font", ("Segoe UI", 10))
            self.option_add("*Button.Font", ("Segoe UI", 10, "bold"))
            # caret color for Entry widgets
            self.option_add("*Entry.insertBackground", TEXT)
        except Exception:
            pass

        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.style.configure(
            "Treeview",
            background=PANEL_BG,
            fieldbackground=PANEL_BG,
            foreground=TEXT,
            rowheight=26,
            borderwidth=0,
        )
        self.style.configure(
            "Treeview.Heading",
            background=SELECT_BG,
            foreground=TEXT,
            font=("Segoe UI", 11, "bold"),
        )
        self.style.configure(
            "TButton",
            background=ACCENT,
            foreground=TEXT,
            font=("Segoe UI", 10, "bold"),
            padding=8,
        )
        self.style.map("TButton", background=[("active", "#4450b0")])
        # Additional ttk styling for a cohesive dark theme
        try:
            self.style.configure("TFrame", background=DARK_BG)
            self.style.configure("TNotebook", background=DARK_BG)
            self.style.configure("TNotebook.Tab", background=SELECT_BG, foreground=TEXT, padding=(6, 4))
            self.style.map("TNotebook.Tab", background=[("selected", PANEL_BG), ("!selected", SELECT_BG)])
            self.style.configure("TEntry", fieldbackground=PANEL_BG, foreground=TEXT)
            self.style.configure("TCombobox", fieldbackground=PANEL_BG, foreground=TEXT)
            self.style.map("Treeview", background=[("selected", SELECT_BG)], foreground=[("selected", SELECT_FG)])
            self.style.configure("Vertical.TScrollbar", background=PANEL_BG, troughcolor=SELECT_BG)
            self.style.configure("Horizontal.TScrollbar", background=PANEL_BG, troughcolor=SELECT_BG)
            # Accent button style to use explicitly where needed
            self.style.configure("Accent.TButton", background=ACCENT, foreground=TEXT, font=("Segoe UI", 10, "bold"), padding=8)
            self.style.map("Accent.TButton", background=[("active", "#4450b0")])
        except Exception:
            pass
        self.style.configure("TLabel", background=DARK_BG, foreground=TEXT, font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground=TEXT)

        self.build_ui()

    def build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, columnspan=3, rowspan=6, sticky="nsew", padx=12, pady=12)

        main_frame = ttk.Frame(notebook)
        guide_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Launcher")
        notebook.add(guide_frame, text="Guide")

        header = ttk.Label(main_frame, text="AlphaModal Launcher", style="Header.TLabel")
        header.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6), padx=(6,0))

        subtitle = ttk.Label(
            main_frame,
            text="Select, inspect, and launch workspace Python scripts through WSL or Modal.",
            wraplength=760,
            justify="left",
        )
        subtitle.grid(row=1, column=0, columnspan=3, sticky="w", padx=(6,0))

        # Search bar
        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=(6,0), pady=(8,6))
        search_frame.columnconfigure(1, weight=1)
        ttk.Label(search_frame, text="Search:").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        try:
            # Ensure caret is visible in dark entry backgrounds
            self.search_entry.configure(insertbackground=TEXT)
        except Exception:
            pass
        self.search_entry.grid(row=0, column=1, sticky="ew", padx=(8,6))
        self.search_entry.bind("<KeyRelease>", lambda e: self.populate_tree(filter_text=self.search_var.get()))
        ttk.Button(search_frame, text="Clear", command=lambda: (self.search_var.set(""), self.populate_tree()), style="Accent.TButton").grid(row=0, column=2, sticky="e")

        # Script tree
        self.tree = ttk.Treeview(main_frame, columns=("description",), show="tree headings", selectmode="browse", height=24)
        self.tree.heading("#0", text="Script")
        self.tree.heading("description", text="Description")
        self.tree.column("#0", width=420, anchor="w")
        self.tree.column("description", width=420, anchor="w")
        self.tree.grid(row=3, column=0, rowspan=6, padx=(6, 6), pady=6, sticky="nsew")

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=3, column=1, rowspan=6, sticky="nsw", padx=(0, 6), pady=6)
        self.tree.configure(yscroll=scrollbar.set)

        # Right column (details + controls)
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=3, column=2, rowspan=6, sticky="nsew", padx=(6,6), pady=6)
        right_frame.grid_rowconfigure(2, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        self.detail_title = ttk.Label(right_frame, text="Select a script to see details", style="Header.TLabel")
        self.detail_title.grid(row=0, column=0, sticky="w")

        self.detail_path = ttk.Label(right_frame, text="Path: -", wraplength=520)
        self.detail_path.grid(row=1, column=0, sticky="w", pady=(6, 6))

        self.detail_text = ScrolledText(right_frame, wrap="word", height=10, background=PANEL_BG, foreground=TEXT, font=("Consolas", 10), relief="flat")
        self.detail_text.grid(row=2, column=0, sticky="nsew")
        self.detail_text.config(state="disabled")

        # Controls
        ctrl_row = 3
        ttk.Label(right_frame, text="Launch mode:").grid(row=ctrl_row+0, column=0, sticky="w", pady=(8,2))
        self.launch_mode_var = tk.StringVar(value="python3 (WSL)")
        self.launch_mode = ttk.Combobox(
            right_frame,
            values=["python3 (WSL)", "modal run", "modal serve", "modal deploy"],
            state="readonly",
            textvariable=self.launch_mode_var,
        )
        self.launch_mode.grid(row=ctrl_row+1, column=0, sticky="ew")
        self.launch_mode.bind("<<ComboboxSelected>>", lambda e: (self.on_launch_mode_change(), self.update_command_preview()))

        ttk.Label(right_frame, text="Entrypoint (modal):").grid(row=ctrl_row+2, column=0, sticky="w", pady=(8,2))
        self.entrypoint_combo = ttk.Combobox(right_frame, values=[], state="disabled")
        self.entrypoint_combo.grid(row=ctrl_row+3, column=0, sticky="ew")
        self.entrypoint_combo.bind("<<ComboboxSelected>>", lambda e: self.update_command_preview())

        ttk.Label(right_frame, text="Extra WSL / modal args:").grid(row=ctrl_row+4, column=0, sticky="w", pady=(8,2))
        self.args_entry = ttk.Entry(right_frame)
        try:
            self.args_entry.configure(insertbackground=TEXT)
        except Exception:
            pass
        self.args_entry.grid(row=ctrl_row+5, column=0, sticky="ew")
        self.args_entry.bind("<KeyRelease>", lambda e: self.update_command_preview())

        # Command preview
        ttk.Label(right_frame, text="Command preview:").grid(row=ctrl_row+6, column=0, sticky="w", pady=(10,2))
        self.preview_text = ScrolledText(right_frame, wrap="none", height=5, background=PANEL_BG, foreground=TEXT, font=("Consolas", 10), relief="flat")
        self.preview_text.grid(row=ctrl_row+7, column=0, sticky="ew")
        self.preview_text.config(state="disabled")

        # Action buttons
        action_frame = ttk.Frame(right_frame)
        action_frame.grid(row=ctrl_row+8, column=0, sticky="ew", pady=(10,4))
        action_frame.columnconfigure((0,1,2), weight=1)
        self.copy_button = ttk.Button(action_frame, text="Copy Command", command=self.copy_command, style="Accent.TButton")
        self.copy_button.grid(row=0, column=0, sticky="ew", padx=(0,6))
        self.open_button = ttk.Button(action_frame, text="Open in WSL", command=self.open_in_wsl, style="Accent.TButton")
        self.open_button.grid(row=0, column=1, sticky="ew", padx=(0,6))
        self.launch_button = ttk.Button(action_frame, text="Launch Selected Script", command=self.on_launch, style="Accent.TButton")
        self.launch_button.grid(row=0, column=2, sticky="ew")

        # Bottom refresh/help
        bottom_frame = ttk.Frame(right_frame)
        bottom_frame.grid(row=ctrl_row+9, column=0, sticky="ew", pady=(8,0))
        bottom_frame.columnconfigure(0, weight=1)
        self.refresh_button = ttk.Button(bottom_frame, text="Refresh Scripts", command=self.populate_tree, style="Accent.TButton")
        self.refresh_button.grid(row=0, column=0, sticky="w")
        ttk.Label(bottom_frame, text="Tip: hover scripts for quick info; use Search to filter.").grid(row=1, column=0, sticky="w", pady=(8,0))

        self.populate_tree()
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<Motion>", self.on_tree_motion)
        self.tree.bind("<Leave>", self.on_tree_leave)

        Tooltip(self.launch_mode, "Choose how to run the script: local WSL python3, or Modal (run/serve/deploy).")
        Tooltip(self.entrypoint_combo, "Select the modal entrypoint function to invoke (if available).")
        Tooltip(self.args_entry, "Enter optional command-line arguments. For 'modal run' put args after -- e.g. -- --port 8000 or just enter named args; the launcher will append them appropriately.")
        Tooltip(self.launch_button, "Launch the selected script using the chosen mode.")
        Tooltip(self.refresh_button, "Reload the list of available Python scripts from the workspace.")

        self.tree_tooltip = Tooltip(self.tree, lambda: "")
        self.tree_tooltip_item = None

        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(2, weight=0)

        guide_text = ScrolledText(guide_frame, wrap="word", background=PANEL_BG, foreground=TOOLTIP_FG, font=("Segoe UI", 10), relief="flat")
        guide_text.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        guide_text.insert("1.0",
            "AlphaModal Launcher Guide\n\n"
            "1) Select a script from the left pane. Scripts are grouped by folder.\n\n"
            "2) Hover over any script title to see a tooltip with its description and usage notes.\n\n"
            "3) Review the title and description on the right. The path shows the relative workspace location.\n\n"
            "4) Optionally enter extra WSL arguments for the script. Example: --model gpt4o --port 8000\n\n"
            "5) Click 'Launch Selected Script' to open the script in WSL using python3.\n\n"
            "6) Use 'Refresh Scripts' after adding or modifying Python files in the workspace.\n\n"
            "7) Tooltips are generated from each script's top comment or module docstring. If a script has no description, add one to improve the launcher.\n\n"
            "This launcher uses WSL instead of Windows virtual environments for better compatibility with your Modal-based workflows.\n\n"
            "Troubleshooting\n"
            "- If WSL cannot be found, make sure WSL is installed and accessible via wsl.exe.\n"
            "- If the script requires a specific Python interpreter inside WSL, adjust the command manually in the code or install python3 in your WSL distro.\n"
            "- Scripts without top-line comments display a generic prompt, so add a summary comment for clarity.\n"
        )
        guide_text.config(state="disabled")
        guide_frame.grid_rowconfigure(0, weight=1)
        guide_frame.grid_columnconfigure(0, weight=1)

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        categories = {}
        for script in self.scripts:
            if hasattr(self, 'search_var') and self.search_var.get().strip():
                q = self.search_var.get().lower()
                if q not in script['title'].lower() and q not in str(script['rel']).lower() and q not in script['description'].lower():
                    continue
            cat = script["category"]
            if cat not in categories:
                categories[cat] = self.tree.insert("", "end", text=cat, open=True)
            self.tree.insert(categories[cat], "end", iid=str(script["rel"]), text=script["title"], values=(script["description"],))

    def on_select(self, event):
        node = self.tree.focus()
        if not node:
            return
        if self.tree.parent(node) == "":
            self.selected_script = None
            self.detail_title.config(text="Select a script to see details")
            self.detail_path.config(text="Path: -")
            self.detail_text.config(state="normal")
            self.detail_text.delete("1.0", "end")
            self.detail_text.config(state="disabled")
            return
        selected = next((s for s in self.scripts if str(s["rel"]) == node), None)
        self.selected_script = selected
        if selected is None:
            return
        self.detail_title.config(text=selected["title"])
        self.detail_path.config(text=f"Path: {selected['rel']}")
        self.detail_text.config(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", selected["description"])
        self.detail_text.config(state="disabled")
        # populate entrypoint selector if available
        eps = selected.get("entrypoints", []) if selected else []
        if eps:
            self.entrypoint_combo.config(state="readonly")
            self.entrypoint_combo["values"] = eps
            try:
                self.entrypoint_combo.set(eps[0])
            except Exception:
                pass
        else:
            self.entrypoint_combo.set("")
            self.entrypoint_combo["values"] = []
            self.entrypoint_combo.config(state="disabled")
        # update entrypoint availability based on launch mode
        self.on_launch_mode_change()
        self.update_command_preview()

    def on_launch(self):
        if not self.selected_script:
            messagebox.showwarning("No script selected", "Please select a script from the list before launching.")
            return
        # map combobox label to internal mode
        mode_label = self.launch_mode_var.get() if hasattr(self, 'launch_mode_var') else 'python3 (WSL)'
        mode_map = {
            'python3 (WSL)': 'python3',
            'modal run': 'modal_run',
            'modal serve': 'modal_serve',
            'modal deploy': 'modal_deploy',
        }
        selected_mode = mode_map.get(mode_label, 'python3')
        entrypoint = None
        if selected_mode == 'modal_run' and self.entrypoint_combo and self.entrypoint_combo.get():
            entrypoint = self.entrypoint_combo.get()

        command = create_command(self.selected_script["path"], self.args_entry.get(), launch_mode=selected_mode, entrypoint=entrypoint)
        try:
            subprocess.Popen(["wsl.exe", "bash", "-lc", command])
            messagebox.showinfo("Launched", f"Script launched ({mode_label}):\n{self.selected_script['rel']}")
        except FileNotFoundError:
            messagebox.showerror("WSL not found", "Could not find wsl.exe. Make sure WSL is installed and available in PATH.")
        except Exception as exc:
            messagebox.showerror("Launch error", f"Unable to launch script:\n{exc}")
        finally:
            self.update_command_preview()

    def update_command_preview(self):
        # Build and display command preview based on current selections
        if not getattr(self, 'selected_script', None):
            txt = "No script selected."
        else:
            mode_label = self.launch_mode_var.get() if hasattr(self, 'launch_mode_var') else 'python3 (WSL)'
            mode_map = {
                'python3 (WSL)': 'python3',
                'modal run': 'modal_run',
                'modal serve': 'modal_serve',
                'modal deploy': 'modal_deploy',
            }
            selected_mode = mode_map.get(mode_label, 'python3')
            entrypoint = None
            if selected_mode == 'modal_run' and self.entrypoint_combo and self.entrypoint_combo.get():
                entrypoint = self.entrypoint_combo.get()
            cmd = create_command(self.selected_script['path'], self.args_entry.get(), launch_mode=selected_mode, entrypoint=entrypoint)
            txt = cmd
        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', 'end')
        self.preview_text.insert('1.0', txt)
        self.preview_text.config(state='disabled')

    def copy_command(self):
        self.preview_text.config(state='normal')
        cmd = self.preview_text.get('1.0', 'end').strip()
        self.preview_text.config(state='disabled')
        if not cmd:
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(cmd)
            messagebox.showinfo('Copied', 'Command copied to clipboard')
        except Exception as e:
            messagebox.showerror('Error', f'Could not copy to clipboard: {e}')

    def open_in_wsl(self):
        # Open a wsl shell and print the command (does not execute automatically)
        if not getattr(self, 'selected_script', None):
            messagebox.showwarning('No script selected', 'Select a script first')
            return
        cmd = self.preview_text.get('1.0', 'end').strip()
        if not cmd:
            messagebox.showwarning('No command', 'No command to open in WSL')
            return
        try:
            subprocess.Popen(['wsl.exe', 'bash', '-lc', f'echo "{cmd}"; bash'], creationflags=0)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to open WSL: {e}')

    def on_launch_mode_change(self):
        # Enable entrypoint selection only for modal_run where entrypoints exist
        mode_label = self.launch_mode_var.get() if hasattr(self, 'launch_mode_var') else 'python3 (WSL)'
        if mode_label == 'modal run' and self.selected_script and self.selected_script.get('entrypoints'):
            self.entrypoint_combo.config(state='readonly')
            if not self.entrypoint_combo.get() and self.selected_script.get('entrypoints'):
                try:
                    self.entrypoint_combo.set(self.selected_script.get('entrypoints')[0])
                except Exception:
                    pass
        else:
            # disable
            try:
                self.entrypoint_combo.config(state='disabled')
            except Exception:
                pass

    def on_tree_motion(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id or item_id == self.tree_tooltip_item:
            return
        if self.tree.parent(item_id) == "":
            self.tree_tooltip.hide()
            self.tree_tooltip_item = None
            return
        script = next((s for s in self.scripts if str(s["rel"]) == item_id), None)
        if not script:
            self.tree_tooltip.hide()
            self.tree_tooltip_item = None
            return
        extra = []
        if script.get('uses_modal'):
            extra.append('Uses Modal: yes')
        if script.get('entrypoints'):
            extra.append('Entrypoints: ' + ', '.join(script.get('entrypoints')))
        extra_txt = ('\n\n' + '\n'.join(extra)) if extra else ''
        tooltip_text = f"{script['title']}\n\n{script['description']}\n\nPath: {script['rel']}{extra_txt}"
        self.tree_tooltip.set_text(tooltip_text)
        self.tree_tooltip.hide()
        self.tree_tooltip.show(event)
        self.tree_tooltip_item = item_id

    def on_tree_leave(self, event):
        self.tree_tooltip.hide()
        self.tree_tooltip_item = None


if __name__ == "__main__":
    all_scripts = get_scripts()
    if not all_scripts:
        messagebox.showerror("No scripts found", "No Python scripts were discovered in the workspace.")
        sys.exit(1)
    app = LauncherApp(all_scripts)
    app.mainloop()

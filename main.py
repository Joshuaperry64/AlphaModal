"""Main entrypoint for the AlphaModal WSL Script Launcher GUI.

Run this file to start the launcher (preferred entrypoint over `launcher.py`).
"""
import sys
from launcher import get_scripts, LauncherApp


def main():
    all_scripts = get_scripts()
    if not all_scripts:
        print("No Python scripts found in the workspace.")
        sys.exit(1)
    app = LauncherApp(all_scripts)
    app.mainloop()


if __name__ == "__main__":
    main()

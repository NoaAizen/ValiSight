#!/usr/bin/env python3
"""Open Thermal Fusion as a desktop window, reusing the running acquisition."""
import fcntl
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
STATE = Path.home() / '.local/state/thermal-fusion'
LOG = STATE / 'desktop-launch.log'
HTTP = int(os.environ.get('HTTP', '8088'))
URL = f'http://127.0.0.1:{HTTP}'
OPENER = build_opener(ProxyHandler({}))


def viewer_ready():
    try:
        with OPENER.open(URL + '/ui', timeout=2) as response:
            data = json.load(response)
        return (isinstance(data, dict) and 'timing' in data
                and isinstance(data.get('cfg'), dict)
                and 'channel' in data['cfg'])
    except (OSError, ValueError):
        return False


def acquisition_running():
    """Do not let run_live.sh interrupt an existing capture or a starting viewer."""
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            if any(Path(os.fsdecode(arg)).name in ('live.py', 'capture.py')
                   for arg in args if arg):
                return True
        except (OSError, ValueError):
            continue
    return False


def ensure_viewer(report):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'startup.lock').open('w') as lock:
        deadline = time.monotonic() + 300
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                report('Another window is starting the system. Waiting...')
                if time.monotonic() > deadline:
                    raise RuntimeError('Startup is still busy. See the launch log.')
                time.sleep(1)
        if viewer_ready():
            return
        if acquisition_running():
            report('A sensor session is already running. Waiting for its viewer...')
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if viewer_ready():
                    return
                time.sleep(1)
            raise RuntimeError('An existing capture or viewer is using the sensors, '
                               'but its interface is unavailable. Finish that session first.')
        report('Starting cameras, radar and AI. This can take a few minutes...')
        with LOG.open('w') as output:
            result = subprocess.run(['bash', str(HERE / 'run_live.sh')],
                                    cwd=HERE, stdout=output, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        if result.returncode or not viewer_ready():
            raise RuntimeError('The sensor system could not start. Check power and USB '
                               'connections, then open the launch log for details.')


def browser_command():
    browser = next((shutil.which(name) for name in
                    ('chromium', 'chromium-browser', 'google-chrome')
                    if shutil.which(name)), None)
    if browser is None:
        raise RuntimeError('Chromium or Google Chrome is required for the desktop window.')
    # This location is accessible to the installed Chromium snap. A dedicated
    # profile keeps the app separate from the user's ordinary browser tabs.
    profile = Path.home() / 'snap/chromium/common/thermal-fusion-app'
    return [browser, f'--app={URL}', f'--user-data-dir={profile}',
            '--class=thermal-fusion', '--no-first-run', '--no-default-browser-check',
            '--window-size=1440,1000']


def install_launchers():
    """Install a menu entry and a desktop shortcut for this checkout."""
    def quote(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace(
            '`', '\\`').replace('$', '\\$') + '"'

    applications = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share'))) / 'applications'
    desktop = Path.home() / 'Desktop'
    if shutil.which('xdg-user-dir'):
        result = subprocess.run(['xdg-user-dir', 'DESKTOP'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            desktop = Path(result.stdout.strip())
    entry = ('[Desktop Entry]\nType=Application\nVersion=1.0\n'
             'Name=Thermal Fusion\nName[he]=מערכת צילום תרמי\n'
             'Comment=Live thermal, visible and radar monitoring\n'
             f'Exec={quote(sys.executable)} {quote(HERE / "desktop_app.py")}\n'
             f'Icon={HERE / "thermal-fusion.svg"}\nTerminal=false\n'
             'Categories=Science;\nStartupNotify=false\n'
             'StartupWMClass=thermal-fusion\n')
    STATE.mkdir(parents=True, exist_ok=True)
    for folder in (applications, desktop):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / 'thermal-fusion.desktop'
        target.write_text(entry)
        target.chmod(0o755)
        print(target)
    if shutil.which('gio'):
        result = subprocess.run(['gio', 'set', str(desktop / 'thermal-fusion.desktop'),
                                 'metadata::trusted', 'true'], capture_output=True)
        if result.returncode:
            print('Desktop shortcut installed. If prompted, choose Allow Launching. '
                  'The application-menu entry can also be used directly.')


def main():
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title('Thermal Fusion')
    root.geometry('540x250')
    root.minsize(460, 230)
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill='both', expand=True)
    ttk.Label(frame, text='Thermal Fusion', font=('', 20, 'bold')).pack(anchor='w')
    status = tk.StringVar(value='Connecting to the sensor system...')
    ttk.Label(frame, textvariable=status, wraplength=460).pack(anchor='w', pady=16)
    progress = ttk.Progressbar(frame, mode='indeterminate')
    progress.pack(fill='x')
    progress.start()
    ttk.Label(frame, text='Closing the viewer leaves acquisition and recording running.',
              wraplength=460).pack(anchor='w', pady=12)
    events = queue.Queue()

    def work():
        try:
            command = browser_command()
            ensure_viewer(lambda text: events.put(('status', text)))
            STATE.mkdir(parents=True, exist_ok=True)
            with (STATE / 'desktop-window.log').open('a') as output:
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                code = child.wait(timeout=3)
                if code:
                    raise RuntimeError('The app window could not open. See desktop-window.log '
                                       'in ' + str(STATE))
            except subprocess.TimeoutExpired:
                pass
            events.put(('done', ''))
        except Exception as exc:
            events.put(('error', str(exc)))

    def show_events():
        try:
            while True:
                kind, text = events.get_nowait()
                if kind == 'done':
                    root.destroy()
                    return
                status.set(text)
                if kind == 'error':
                    progress.stop()
                    progress.pack_forget()
                    ttk.Button(frame, text='Open logs', command=lambda: subprocess.Popen(
                        ['xdg-open', str(STATE)])).pack(side='left')
                    ttk.Button(frame, text='Close', command=root.destroy).pack(side='right')
        except queue.Empty:
            pass
        root.after(100, show_events)

    threading.Thread(target=work, daemon=True).start()
    root.after(100, show_events)
    root.mainloop()


if __name__ == '__main__':
    if sys.argv[1:] == ['--install']:
        install_launchers()
    else:
        main()

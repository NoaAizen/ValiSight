# ValiSight

mmWave radar (TI IWR1843) + OpenMV N6 (PAG7936 RGB / FLIR Lepton 3.5 thermal)
fusion rig, running on a Jetson Orin at `~/valisigth`. Everything belonging to
the project lives under this one directory, and it is what the dev container
mounts at `/workspace`.

## Layout

    src/            host-side code, runs on the Jetson (CPython)
      cfg/          IWR1843 chirp configs (odom_b .. odom_f sweeps)
      stock_iwr1843.cfg
    board/          MicroPython — copied to the OpenMV N6, does NOT run on the host
    docker/         Dockerfile + run.sh for the dev container
    data/
      recordings/   capture sessions, one dir per run (see src/recorder.py)
      captures/     stills and Lepton diagnostic dumps (captures/diag)
    shared/         papers and notes; exported to Windows over Samba
    reference/      read-only material: OpenMV firmware sources that explain the
                    Lepton path (lepton.c, vospi.c, py_csi.c, LEPTON_OEM.h) and
                    the bring-up report
    env/            virtualenv (unused so far: pip + setuptools only)
    tools/          jetson-containers (third-party build toolkit)
    attic/          dead leftovers kept only until they are confirmed junk:
                    0-byte torch/torchvision .whl stubs, test_.txt, and
                    scratch-bringup/ — one-off probe scripts from the bring-up
                    sessions, rescued off /tmp before it is wiped

`src/` and `board/` deliberately hold two copies of `radar_gate.py` and
`radar_classify_n6.py` (`board/sdcard_*.py`). They are pure modules that run in
both places; the `board/` copies are a snapshot of what is actually flashed, so
they change only when the board is reflashed.

## Running

Everything host-side assumes it is executed from `src/`, and resolves `data/`
relative to its own file, so absolute paths are never needed.

    cd src
    python3 live_server.py --record        # live view + recording
    python3 run_system.py                  # radar-only pipeline
    python3 radar_replay.py --list         # offline scoring over data/recordings
    python3 lepton_diag.py capture         # thermal diagnostics -> data/captures/diag

Container:

    docker/run.sh                          # shell in /workspace
    docker/run.sh python3 src/run_system.py

`run.sh` mounts this directory — it derives the workspace from its own location,
so the project can be moved or renamed without editing it (override with
`VALISIGHT_WORKSPACE`). It passes through every `/dev/ttyACM*` and `/dev/video*`
it finds, plus `/dev/serial/by-id` — resolve the radar and the N6 through the
by-id symlinks, because `ttyACM` numbering swaps between replugs.

## Notes

- `shared/` is the Samba share `[valisight_shared]`, reachable from Windows.
  `smb.conf` still names the old path `/home/valisigth/valisight_shared`, which
  is now a symlink here. To make it permanent:

      sudo sed -i 's#/home/valisigth/valisight_shared#/home/valisigth/valisigth/shared#' /etc/samba/smb.conf
      sudo systemctl reload smbd && rm ~/valisight_shared

- `tools/jetson-containers` is the real directory now. `/usr/local/bin/jetson-containers`
  and `/usr/local/bin/autotag` are root-owned symlinks into `~/jetson-containers`,
  so that path is kept as a symlink here and both commands still work. To drop
  the shim:

      sudo ln -sfn ~/valisigth/tools/jetson-containers/jetson-containers /usr/local/bin/jetson-containers
      sudo ln -sfn ~/valisigth/tools/jetson-containers/autotag /usr/local/bin/autotag
      rm ~/jetson-containers

- `nomachine.deb` and `snapd_24724.*` are left in `$HOME`: installers for the
  box, unrelated to ValiSight.

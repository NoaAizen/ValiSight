# Building the firmware with the fusion module

The OpenMV firmware tree is not vendored here. It is 2.6 GB of upstream code, of
which eleven lines are ours, and those eleven are in
`openmv-fusion-module.patch`.

```sh
git clone --recursive https://github.com/openmv/openmv.git
cd openmv
git checkout be63fec4fba63accdb47f2c4ffbf84017555e538
git apply /path/to/openmv-integration/openmv-fusion-module.patch

# One source of truth for the C: the firmware compiles the same files the host
# does, by symlink rather than by copy. A copy drifts, and it drifts silently -
# the host tests keep passing against a file the firmware no longer uses.
ln -s ../../src/fusion.c    modules/fusion.c
ln -s ../../src/fusion.h    modules/fusion.h
ln -s ../../src/py_fusion.c modules/py_fusion.c

export PATH=/path/to/arm-gnu-toolchain-14.3.rel1/bin:$PATH
make TARGET=OPENMV_N6 -j$(nproc) firmware
```

`SRC_USERMOD += $(wildcard modules/*.c)` picks the module up on its own, so the
patch only has to switch it on.

## What the patch does

`boards/OPENMV_N6/board_config.mk` sets `MICROPY_PY_FUSION = 1`.

`common/micropy.mk` turns that into three defines:

| define | why |
|---|---|
| `MICROPY_PY_FUSION=1` | compiles `py_fusion.c` in |
| `FUSION_USE_MP_ALLOC=1` | buffers come from `uma_calloc(UMA_FAST\|UMA_PERSIST\|UMA_MAYBE)` rather than libc, so they land in internal SRAM and are traced through the owning object instead of leaking across a soft reset |
| `FUSION_ENABLE_HOMOGRAPHY=0` | compiles out the only floating-point function in the library |

That last one matters more than it looks. OpenMV builds with
`-fsingle-precision-constant -Wdouble-promotion -Werror`: the M55 has a
single-precision FPU, so every `double` becomes software emulation.
`fusion_set_homography()` is setup-only and never runs on the board - the warp
table always arrives from calibration - so compiling it out makes the target
build integer end to end. Verified rather than assumed:

```
$ arm-none-eabi-nm -u build/OPENMV_N6/lib/micropython/modules/fusion.o
         U __aeabi_ldivmod
         U memcpy
         U memset
```

No `__aeabi_d*`, no `__aeabi_f*`. The only helper is 64-bit integer division,
which the radiometric path needs. The claim in `fusion.h`'s header comment is
enforced by the compiler, not by discipline.

## Footprint

| object | text | bss |
|---|---|---|
| `fusion.o` | 8228 B | 3076 B (four 256x3 palettes) |
| `py_fusion.o` | 2377 B | 0 |

`firmware.bin` comes to 2,000,472 bytes, 54.5% of `FLASH_TEXT`.

## Toolchain notes

Environment obstacles hit on an aarch64 host, none of them bugs in this code:

- The OpenMV SDK has no aarch64 build (404). A version stamp in
  `~/openmv-sdk-1.6.0/` plus an own toolchain is enough for `make firmware`.
- `check_toolchain.mk` rejects GCC 14.2 for the M55; 14.3.rel1 passes.
- ARM `CFLAGS` leak into `mpy-cross`, which is a host tool. Build it separately
  with `env -u CFLAGS`.
- `STM32_SigningTool_CLI` is only needed for the bootloader (`-t fsbl`).

**Flashing is blocked on x86_64**: `stcubeprog` and `stedgeai` ship for x86_64
only. `tools/pydfu.py` is pure Python, but the N6 boots from external flash and
pydfu targets internal - not something to try on a board you cannot recover.

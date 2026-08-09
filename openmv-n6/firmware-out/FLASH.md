# Flashing this build

Built on the Jetson 2026-08-09 from `src/fusion.c` @ 1457 lines (the merged tree,
including the temporal filters). Checksums in `SHA256SUMS`.

| file | size | what it is |
|---|---|---|
| `openmv.bin` | 2,525,480 B | **the one you flash** - bootloader + firmware + romfs, one image |
| `firmware.bin` | 2,001,192 B | firmware alone, 54.53% of FLASH_TEXT. For reference/size tracking |
| `bootloader.bin` | 46,316 B | unchanged since 2026-08-04; already inside `openmv.bin` |

## What is in it

`fusion.o` 9260 B text + 3076 B bss, `py_fusion.o` 2377 B. Against a 3584 KB
FLASH_TEXT that is noise - the module is not what fills the part.

Verified integer end to end, by the linker rather than by discipline:

```
$ arm-none-eabi-nm -u build/OPENMV_N6/lib/micropython/modules/fusion.o
         U __aeabi_ldivmod        <- 64-bit integer division, the radiometric path
         U memcpy
         U memset
         U uma_calloc             <- FUSION_USE_MP_ALLOC=1, buffers in internal SRAM
         U uma_free
```

No `__aeabi_d*`, no `__aeabi_f*`. `FUSION_ENABLE_HOMOGRAPHY=0` compiles out the
one floating-point function, which is setup-only and never runs on the board.

`fusion_process`, `fusion_init`, `fusion_thermal_temporal`, `fusion_temp_at` and
`py_fusion_type` are all present in `firmware.elf`, so the MicroPython module is
exposed and not merely compiled.

## Two commands. Prefer the second.

`openmv.bin` is not a special format: `port_config.mk:167-171` pads
`bootloader.bin` with 0xFF up to `OMV_FIRM_ADDR - OMV_FIRM_BASE` (0x80000) and
concatenates `firmware.bin` onto it. So the choice is only about how much of the
flash you are willing to have open at once.

**Full image** - the tree's own `deploy` target, `port_config.mk:194-197`:

```
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG ap=1 \
    -el <SDK>/stcubeprog/bin/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr \
    -w openmv.bin 0x70000000 -hardRst
```

**Firmware only** - same tool, same loader, starts above the bootloader:

```
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG ap=1 \
    -el <SDK>/stcubeprog/bin/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr \
    -w firmware.bin 0x70080000 -hardRst
```

`0x70000000` is `OMV_FIRM_BASE` and `0x70080000` is `OMV_FIRM_ADDR`, both from
`boards/OPENMV_N6/board_config.mk:8-9`. The external loader is `OMV_PROG_STLDR`
from line 23 - the N6 boots from external QSPI flash, which is why a plain DFU
write against internal flash is not the right tool.

The trade is real and worth stating: the full image guarantees a matched
bootloader/firmware pair, and is the only option if the two ever diverge.
Firmware-only keeps whatever bootloader is already on the board, so if a write
goes wrong there is still something there to recover through. Note that nobody
has verified what bootloader the board is actually carrying - `bootloader.bin`
here is from 2026-08-04 and was not rebuilt.

## How bad can it get

Not as bad as the tone of `openmv-integration/README.md` implies, and the reason
is specific: **neither command touches option bytes.** They are `-w <file> <addr>`
and nothing else - no `-ob`, no `-rdp`. RDP level 2 is the one genuinely
irreversible operation on an STM32, since it disables the debug port for good,
and it is nowhere near this path.

Recovery also does not depend on what was just written. The ST-Link addresses the
core directly and the `.stldr` runs from RAM to drive the QSPI part, so a corrupt
or interrupted write yields a board that will not boot, which you fix by
connecting the probe and writing again.

What can actually kill the board is physical: wrong voltage or wiring on the SWD
header, or losing power mid-erase. Have the probe attached and talking before
starting - `-c port=SWD` needs it anyway, so if it is not, the write must not
begin.

`boot/src/common/dfu.c` exists, so the bootloader does speak DFU and the USB path
through OpenMV IDE is a real third option - lowest risk of the three, because it
needs no probe and cannot reach the bootloader region at all. Untested here.

## After flashing

The board should enumerate as before. Then, in order:

1. `csi.devices()` returns the three CIDs - nothing about the sensors changed.
2. `import fusion` works. That is the whole point of this build and is the first
   thing that tells you the module survived the link and the flash.
3. Time `fusion_process()` on the board. The open question this build exists to
   answer: the Jetson runs it scalar in 8.80 ms synthetic / 10.5 ms on real data,
   and the M55 estimate is 40-80 ms against the Lepton's 113 ms cycle. If it fits,
   **Helium (M5) optimisation is unnecessary and comes off the roadmap.** If it
   does not, that is the number that justifies the work.

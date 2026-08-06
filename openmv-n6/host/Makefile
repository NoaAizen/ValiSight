CC      ?= cc
# _POSIX_C_SOURCE: strict c99 hides clock_gettime, which the bench path needs
CFLAGS  ?= -std=c99 -D_POSIX_C_SOURCE=199309L -O2 -Wall -Wextra -Wno-unused-parameter -I../src
LDFLAGS ?=

SRC := main.c ../src/fusion.c
BIN := fuse

all: $(BIN) libfusion.so

$(BIN): $(SRC) ../src/fusion.h
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

# Shared library so the live viewer can drive the exact same C the firmware runs.
libfusion.so: ../src/fusion.c ../src/fusion.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ ../src/fusion.c

# Same sources under the sanitizers - the reference path must be clean before
# any Helium version is written against it.
asan: $(SRC) ../src/fusion.h
	$(CC) $(CFLAGS) -g -fsanitize=address,undefined -o $(BIN)-asan $(SRC) $(LDFLAGS)

test: $(BIN) asan
	python3 synth.py
	./$(BIN)-asan synth/y.raw synth/thermal.raw synth/out.ppm --warp synth/warp.lut
	# the row repair and the radiometric queries under the sanitizers too - they
	# index the warp table and the thermal frame from user-supplied coordinates
	./$(BIN)-asan synth/y.raw synth/thermal_dead.raw synth/out.ppm --warp synth/warp.lut \
		--range 20,45 --emissivity 0.1 --reflected 18 --stats --probe 400,125
	./$(BIN)-asan synth/y.raw synth/thermal_hot.raw synth/out.ppm --warp synth/warp.lut \
		--stats --probe 0,0
	./$(BIN) synth/y.raw synth/thermal.raw synth/out.ppm --warp synth/warp.lut --bench 20
	python3 verify.py

# Real recorded frames rather than synthetics. Kept out of `test` because it
# needs captures on disk, but it is the only check that can catch the detector
# tracking the scene instead of the defect - which the first version did.
CAPTURES ?= $(wildcard ../captures/*)

real: $(BIN)
	python3 verify_real.py $(CAPTURES)

clean:
	rm -f $(BIN) $(BIN)-asan libfusion.so
	rm -rf synth

.PHONY: all asan test real clean

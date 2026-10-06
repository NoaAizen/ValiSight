PYTHON ?= python3
PROJECT := openmv-n6
export PYTHON
.DEFAULT_GOAL := help

.PHONY: help doctor check build test test-python test-scripts test-c test-real clean

help:
	@echo "make doctor       Check host dependencies and local artifacts (read-only)"
	@echo "make check        Check Python, shell and notebook source syntax"
	@echo "make build        Build the host C executable and shared library"
	@echo "make test         Run all offline checks, including sanitizers"
	@echo "make test-python  Run pytest (PyTorch checks skip when unavailable)"
	@echo "make test-scripts Run offline viewer, radar and calibration suites"
	@echo "make test-real    Verify the checked-in recorded frames"
	@echo "make clean        Remove generated host binaries and synthetic fixtures"

doctor:
	$(PYTHON) scripts/doctor.py

check:
	$(PYTHON) scripts/check_project.py

build:
	$(MAKE) -C $(PROJECT)/host all

test: check test-c test-python test-scripts test-real

test-python:
	$(PYTHON) -m pytest

# These suites have main functions or top-level checks, not pytest test cases.
test-scripts: build
	$(PYTHON) $(PROJECT)/tools/tests/test_chirp.py
	$(PYTHON) $(PROJECT)/tools/tests/test_heatmaps.py
	$(PYTHON) $(PROJECT)/tools/tests/test_radar.py
	$(PYTHON) $(PROJECT)/tools/tests/test_radar_overlay.py
	$(PYTHON) $(PROJECT)/tools/tests/test_live.py
	$(PYTHON) $(PROJECT)/tools/calib/test_calib.py
	$(PYTHON) $(PROJECT)/tools/calib/test_radar_extrinsics.py

test-c:
	$(MAKE) -C $(PROJECT)/host test

# Use the checked-in fixture, so local capture directories cannot change CI.
test-real:
	$(MAKE) -C $(PROJECT)/host real CAPTURES=../captures/handwave3

clean:
	$(MAKE) -C $(PROJECT)/host clean

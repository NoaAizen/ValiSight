"""The public surface: that it resolves, that it is described, and that it stays cheap.

Two other repositories import this package, so its surface is a contract rather
than a convenience. These tests hold the three properties that contract rests
on: every name works, every name says what it is for, and naming one costs only
what it has to.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import mapinit
from mapinit.api import AREAS, PUBLIC_API, describe, exports_by_area

#: Third-party stacks that must not arrive merely because something was named.
HEAVY = ("pyproj", "rasterio", "shapely")


# -- the surface resolves ---------------------------------------------------


@pytest.mark.parametrize("export", PUBLIC_API, ids=lambda export: export.name)
def test_every_catalogued_name_resolves(export):
    """A catalogued name that no longer exists is documentation of something deleted."""
    assert getattr(mapinit, export.name) is not None


@pytest.mark.parametrize("export", PUBLIC_API, ids=lambda export: export.name)
def test_every_name_comes_from_the_module_it_claims(export):
    """The module column is what a reader greps by; a wrong one sends them nowhere."""
    from importlib import import_module

    module = import_module(export.module, "mapinit")
    assert hasattr(module, export.name)


def test_all_is_built_from_the_catalogue():
    """__all__ is generated, not maintained: the two cannot drift apart."""
    catalogued = {export.name for export in PUBLIC_API}
    assert catalogued <= set(mapinit.__all__)


def test_no_name_is_catalogued_twice():
    names = [export.name for export in PUBLIC_API]
    assert len(names) == len(set(names))


def test_every_export_sits_in_a_declared_area():
    known = {name for name, _ in AREAS}
    assert {export.area for export in PUBLIC_API} <= known


def test_every_area_has_something_in_it():
    """An empty area is a heading with nothing under it."""
    for area, exports in exports_by_area().items():
        assert exports, f"area {area!r} is declared but exports nothing"


# -- the surface is described ----------------------------------------------


@pytest.mark.parametrize("export", PUBLIC_API, ids=lambda export: export.name)
def test_every_name_says_what_it_is_and_what_it_is_for(export):
    """A bare name is not a surface. This is the property the registry exists for."""
    assert export.summary.strip(), f"{export.name} has no summary"
    assert export.use.strip(), f"{export.name} has no use"
    assert export.summary.strip().endswith("."), f"{export.name}: summary is not a sentence"


def test_describe_covers_every_export():
    text = describe()
    for export in PUBLIC_API:
        assert export.name in text


def test_describe_can_be_narrowed_to_one_area():
    text = describe("navigation")
    assert "DeadReckoner" in text
    assert "MapInitializer" not in text


def test_describe_refuses_an_unknown_area_and_names_the_real_ones():
    with pytest.raises(KeyError) as excinfo:
        describe("nagivation")
    assert "navigation" in str(excinfo.value)


# -- the surface stays cheap ------------------------------------------------


def _modules_after(statement: str) -> set:
    """Which heavy modules a fresh interpreter ends up holding after this code."""
    source = (
        "import sys\n"
        f"{statement}\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    return {name for name in out.stdout.strip().split(",") if name}


def test_importing_the_package_pulls_in_nothing_heavy():
    """Perception vendors the health core; it must not inherit a raster stack."""
    assert _modules_after("import mapinit") == set()


def test_naming_a_navigation_export_pulls_in_nothing_heavy():
    """Drift and heading are pure arithmetic and must stay importable anywhere."""
    assert _modules_after("import mapinit; mapinit.DeadReckoner; mapinit.HeadingMatcher") == set()


def test_naming_a_health_export_pulls_in_nothing_heavy():
    assert _modules_after("import mapinit; mapinit.Check; mapinit.InitStage") == set()


def test_naming_a_geo_export_does_pull_its_dependency():
    """The other half of the same property: laziness defers, it does not remove."""
    assert "pyproj" in _modules_after("import mapinit; mapinit.GeoidModel")


# -- failure modes ----------------------------------------------------------


def test_an_unknown_attribute_points_at_the_catalogue():
    """The error is a consumer's first encounter with the surface; make it useful."""
    with pytest.raises(AttributeError) as excinfo:
        mapinit.SolveEverything
    assert "describe()" in str(excinfo.value)


def test_dir_lists_the_surface():
    """What a REPL and an editor complete against."""
    assert "DeadReckoner" in dir(mapinit)
    assert "MapInitializer" in dir(mapinit)


# -- the subpackages agree with the top level -------------------------------


@pytest.mark.parametrize("subpackage", ["mapinit.nav", "mapinit.calibration", "mapinit.geo"])
def test_subpackage_exports_are_all_catalogued(subpackage):
    """Two ways in, one contract. A name reachable by either must be described."""
    from importlib import import_module

    module = import_module(subpackage)
    catalogued = {export.name for export in PUBLIC_API}
    assert set(module.__all__) <= catalogued


# -- reaching for a half that is not installed ------------------------------


def test_a_missing_extra_names_the_extra_not_a_third_party_module():
    """Half this package installs with nothing, so this is an ordinary accident.

    A caller who asked for GeoidModel never asked for pyproj, and telling them
    pyproj is missing makes them go and install the wrong thing.
    """
    source = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name.split('.')[0] == 'pyproj' else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import mapinit\n"
        "try:\n"
        "    mapinit.GeoidModel\n"
        "except ImportError as exc:\n"
        "    print(exc)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    assert "'geo' extra" in out.stdout
    assert "pip install" in out.stdout


def test_pure_areas_declare_no_extra():
    """health and navigation must stay installable with nothing."""
    from mapinit.api import extra_for

    assert extra_for("Check") is None
    assert extra_for("DeadReckoner") is None
    assert extra_for("GeoidModel") == "geo"
    assert extra_for("solve_imu_camera") == "calibration"

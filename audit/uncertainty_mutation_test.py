"""Mutation harness: prove the uncertainty auditor and unit tests bite.

For each mutation, the module + auditor + unit test are copied into a
throwaway temp dir, the mutation is applied TO THE COPY ONLY, and both the
auditor and the unit tests are run against it. A mutation is "caught" when
either exits nonzero. A surviving mutation means the check that should have
caught it is theater — the harness reports it by name and exits nonzero.

The real tree is never modified: this file only ever writes under
tempfile.mkdtemp(). It also verifies each mutation actually changed the
source (a substitution that no longer matches is a harness bug, reported as
such rather than counted as a kill).

Run:  python3 audit/uncertainty_mutation_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = os.path.join(ROOT, "core", "fusion", "uncertainty.py")
AUDITOR = os.path.join(ROOT, "audit", "uncertainty_auditor.py")
UNIT_TEST = os.path.join(ROOT, "tests", "test_uncertainty.py")

# name -> (exact string in uncertainty.py, replacement)
MUTATIONS = {
    "remove_glint_penalty":
        ("GLINT_SIGMA_PENALTY = 1.5", "GLINT_SIGMA_PENALTY = 0.0"),
    "remove_static_penalty":
        ("STATIC_SIGMA_PENALTY = 0.9", "STATIC_SIGMA_PENALTY = 0.0"),
    "zero_clutter_term":
        ("CLUTTER_SIGMA_WEIGHT = 0.5", "CLUTTER_SIGMA_WEIGHT = 0.0"),
    "break_inverse_variance_sigma":
        ("sigma = math.sqrt(1.0 / total_precision)",
         "sigma = sum(e.sigma for e in ests) / len(ests)"),
    "plain_mean_score":
        ("score = weighted_sum / total_precision",
         "score = sum(e.score for e in ests) / len(ests)"),
    "hardwire_contributing_to_thermal":
        ("contributing = best.sensor", "contributing = THERMAL"),
}


def _build_sandbox(tmp, mutation=None):
    """Lay out core/fusion/uncertainty.py + auditor + test under tmp."""
    pkg = os.path.join(tmp, "core", "fusion")
    os.makedirs(pkg)
    for d in (os.path.join(tmp, "core"), pkg):
        with open(os.path.join(d, "__init__.py"), "w") as fh:
            fh.write("")
    with open(MODULE) as fh:
        src = fh.read()
    if mutation is not None:
        old, new = MUTATIONS[mutation]
        if old not in src:
            raise RuntimeError("mutation %r no longer matches the source; "
                               "update MUTATIONS" % mutation)
        src = src.replace(old, new)
        assert src != open(MODULE).read()
    with open(os.path.join(pkg, "uncertainty.py"), "w") as fh:
        fh.write(src)
    shutil.copy(AUDITOR, os.path.join(tmp, "uncertainty_auditor.py"))
    shutil.copy(UNIT_TEST, os.path.join(tmp, "test_uncertainty.py"))


def _run(tmp):
    """Run auditor + unit tests inside tmp. Returns (auditor_rc, tests_rc)."""
    env = dict(os.environ, PYTHONPATH=tmp)
    auditor = subprocess.run([sys.executable, "uncertainty_auditor.py"],
                             cwd=tmp, env=env, capture_output=True)
    tests = subprocess.run([sys.executable, "-m", "unittest", "-q",
                            "test_uncertainty"],
                           cwd=tmp, env=env, capture_output=True)
    return auditor.returncode, tests.returncode


def main():
    # Baseline: the unmutated copy must pass both, or every "kill" below
    # would be meaningless noise.
    tmp = tempfile.mkdtemp(prefix="unc_baseline_")
    try:
        _build_sandbox(tmp)
        a_rc, t_rc = _run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if a_rc or t_rc:
        print("BASELINE BROKEN: unmutated copy fails (auditor rc=%d, tests rc=%d)"
              % (a_rc, t_rc))
        return 2
    print("baseline: unmutated copy passes auditor and tests")

    survivors = []
    for name in MUTATIONS:
        tmp = tempfile.mkdtemp(prefix="unc_mut_")
        try:
            _build_sandbox(tmp, mutation=name)
            a_rc, t_rc = _run(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        caught_by = ([] if not a_rc else ["auditor"]) + ([] if not t_rc else ["tests"])
        if caught_by:
            print("KILLED    %-36s (caught by %s)" % (name, ", ".join(caught_by)))
        else:
            survivors.append(name)
            print("SURVIVED  %-36s <-- this check is theater" % name)

    print("%d/%d mutations caught" % (len(MUTATIONS) - len(survivors), len(MUTATIONS)))
    if survivors:
        print("surviving mutations: %s" % ", ".join(survivors))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

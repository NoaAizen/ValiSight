#!/usr/bin/env python3
"""The catalogue of what this package offers a consumer, and what each piece does.

Two other teams import this code: perception, which trains on radar and thermal
and needs the rig's attitude and the frames it lives in, and fusion, which needs
the geometry that turns a detection into a place. Neither should have to read
``mapinit.geo.dem`` to discover that ``GroundEstimate`` exists, and neither
should discover it by exception when a refactor moves it.

**This module is the source of truth, not a description of one.** ``__init__``
builds its ``__all__`` and its lazy-import table from the tuple below, so a name
cannot become public without a summary saying what it is and a note saying what
it is for. The reverse also holds: a name listed here that no longer resolves
fails a test rather than surviving as documentation of something deleted.

**What is deliberately absent.** Solvers' internal helpers, the stage subclasses
themselves, and anything under a leading underscore. Exporting them would freeze
today's decomposition as a contract and make every refactor a breaking change
for two other repositories. When a consumer needs one of them, it gets added
here with a reason, which is a smaller conversation than un-exporting it later.

Read it from a terminal with::

    python -c "import mapinit; print(mapinit.describe())"
    python -c "import mapinit; print(mapinit.describe('navigation'))"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: Areas, in the order a newcomer should meet them, each with what it covers.
AREAS: Tuple[Tuple[str, str], ...] = (
    ("health", "Validation, staging and reporting. The core other repos vendor."),
    ("map", "Map-relative initialization: the one object, and its context."),
    ("geo", "Vertical datum, terrain, buildings, and tiled prior resolution."),
    ("view", "What the map says should be in frame from a pose."),
    ("calibration", "Solving the rig's fixed geometry, and validating it can be solved."),
    ("navigation", "How a fix decays between corrections, and how heading is recovered."),
)


@dataclass(frozen=True)
class Export:
    """One public name, where it lives, and why a consumer would want it."""

    name: str
    module: str
    area: str
    #: What the thing is, in one line.
    summary: str
    #: What a consumer does with it. Names the caller where there is a known one.
    use: str

    def __str__(self) -> str:
        return f"{self.name:<32} {self.summary}\n{'':<32} -> {self.use}"


PUBLIC_API: Tuple[Export, ...] = (
    # -- health ------------------------------------------------------------
    Export(
        "Check", ".check", "health",
        "One validation, carrying what was measured against what was expected.",
        "Vendored by perception into perception/health/. Return one instead of a "
        "bool so a failure report says which assumption broke, not just that one did.",
    ),
    Export(
        "CheckFailed", ".check", "health",
        "Raised when a check a stage cannot continue without does not pass.",
        "Catch it to distinguish a refused start from a crash.",
    ),
    Export(
        "InitStage", ".stage", "health",
        "Base class for one step of a guarded startup sequence.",
        "Subclass to add a stage; the runner handles ordering, skipping and reporting.",
    ),
    Export(
        "StageResult", ".stage", "health",
        "What one stage produced: its checks, its outputs, and whether it ran.",
        "Read it to find which stage refused and on what evidence.",
    ),
    Export(
        "StageStatus", ".stage", "health",
        "Whether a stage passed, failed, or was skipped with a stated reason.",
        "A skip is not a pass. Branch on this rather than on truthiness.",
    ),
    Export(
        "InitializationPipeline", ".runner", "health",
        "Runs stages in order and collects every check into one report.",
        "Build one with your own stages to reuse the machinery outside map init.",
    ),
    Export(
        "InitReport", ".runner", "health",
        "Every check from every stage of one run, passed and failed alike.",
        "Log it whole. A report that only lists failures hides which guards ran.",
    ),
    Export(
        "CalibrationConstraint", ".calibration.constraints", "health",
        "One source of information about the rig, able to validate itself.",
        "Subclass to add a calibration input without touching any stage or runner.",
    ),
    Export(
        "Observation", ".calibration.constraints", "health",
        "One measurement contributed to a calibration, with its sigma.",
        "The sigma is what lets observations of different kinds be weighted "
        "against each other; do not pass measurements without one.",
    ),

    # -- map ---------------------------------------------------------------
    Export(
        "MapInitializer", ".map_initializer", "map",
        "The one object map-relative initialization is driven through.",
        "Construct with a latitude and longitude, call run() for a guarded start, "
        "priors() for the layers covering the point. Construction touches no disk.",
    ),
    Export(
        "InitContext", ".context", "map",
        "The inputs one initialization run works from.",
        "Build directly only when driving a pipeline yourself; MapInitializer "
        "makes one for you.",
    ),
    Export(
        "GLOBAL_GEOID_BOUND_M", ".context", "map",
        "Largest EGM2008 undulation anywhere on Earth, in metres.",
        "The plausibility bound the geoid guard enforces regardless of location.",
    ),

    # -- geo ---------------------------------------------------------------
    Export(
        "GeoidModel", ".geo.geoid", "geo",
        "EGM2008 undulation lookup: the bridge between ellipsoidal and orthometric height.",
        "Use for any conversion between what GNSS reports and what the map says. "
        "Raises at construction rather than returning a quiet fallback.",
    ),
    Export(
        "GeoidGridUnavailable", ".geo.geoid", "geo",
        "Raised when PROJ has no real vertical shift grid to work from.",
        "Catch it. The alternative is a silent ~20 m height offset in Israel that "
        "looks exactly like a calibration error.",
    ),
    Export(
        "EGM2008_GRID_NAME", ".geo.geoid", "geo",
        "Filename of the vertical shift grid this package expects.",
        "Use it to check an install or to name the file in a fetch script.",
    ),
    Export(
        "EGM2008_GRID_URL", ".geo.geoid", "geo",
        "Where to fetch that grid from.",
        "Quote it in the error message when an install is missing the grid.",
    ),
    Export(
        "DemSampler", ".geo.dem", "geo",
        "Terrain heights from GLO-30, point by point or in bulk.",
        "sample_many() reads the covering window once; use it for ray marching "
        "rather than looping over the single-point call.",
    ),
    Export(
        "GroundEstimate", ".geo.dem", "geo",
        "Ground elevation under a point, with the local plane fitted around it.",
        "The ground plane a detection's height is measured against.",
    ),
    Export(
        "EgoAltitudePrior", ".geo.dem", "geo",
        "Our own altitude, in both height systems, with its uncertainty.",
        "A prior on the rig's height, not a measurement of it. 30 m posting "
        "resolves nothing at obstacle scale.",
    ),
    Export(
        "DemUnavailable", ".geo.dem", "geo",
        "Raised when terrain cannot be sampled where it was asked for.",
        "Catch it rather than accepting a fallback elevation.",
    ),
    Export(
        "DependencyMissing", ".geo.dem", "geo",
        "Raised when the raster stack is not installed.",
        "Separated from DemUnavailable on purpose: it says the install is wrong, "
        "not that the map is.",
    ),
    Export(
        "ScaleNotSupported", ".geo.dem", "geo",
        "Raised when a request is finer than the data can answer.",
        "Guards against reading obstacle-scale structure out of a 30 m posting.",
    ),
    Export(
        "BuildingLayer", ".geo.buildings", "geo",
        "Building footprints for an area, queryable by position and range.",
        "The source of the vertical edges a heading match runs against.",
    ),
    Export(
        "Building", ".geo.buildings", "geo",
        "One footprint, its height, and where that height came from.",
        "Check the height source before trusting a height: most footprints in the "
        "cache have none published.",
    ),
    Export(
        "HeightSource", ".geo.buildings", "geo",
        "Whether a building's height was published, inferred, or defaulted.",
        "Branch on it rather than treating every height as equally known.",
    ),
    Export(
        "resolve_height", ".geo.buildings", "geo",
        "Pick a building's height from what the record offers, and say which it used.",
        "Use it instead of reading the height field directly.",
    ),
    Export(
        "BasePriorDataProvider", ".geo.providers", "geo",
        "Where prior layers are fetched from.",
        "Subclass to serve priors from your own store; the pipeline takes any one.",
    ),
    Export(
        "LocalFilePriorProvider", ".geo.providers", "geo",
        "Priors from files on disk. The offline default.",
        "What a field rig runs on: no network, no surprise.",
    ),
    Export(
        "DatabasePriorProvider", ".geo.providers", "geo",
        "Priors from a database rather than the filesystem.",
        "For a host that already holds the layers.",
    ),
    Export(
        "PriorPaths", ".geo.providers", "geo",
        "The resolved paths of the layers covering one point.",
        "Returned by MapInitializer.priors(); annotate what you hold with it.",
    ),
    Export(
        "TileBounds", ".geo.tiles", "geo",
        "The geographic extent one tile covers.",
        "Use to test coverage before a sample, rather than catching the miss.",
    ),
    Export(
        "TileNotFound", ".geo.tiles", "geo",
        "Raised when no cached tile covers the requested point.",
        "Catch it. Returning a neighbouring tile moves a position without "
        "failing a test.",
    ),
    Export(
        "AmbiguousTiles", ".geo.tiles", "geo",
        "Raised when more than one tile claims a point.",
        "A duplicated or mislabelled cache, caught before it silently picks one.",
    ),
    Export(
        "select_tile", ".geo.tiles", "geo",
        "The tile covering a point, or an exception saying why none does.",
        "The lookup behind every terrain and vector sample.",
    ),
    Export(
        "parse_tile_bounds", ".geo.tiles", "geo",
        "Read a tile's extent out of its filename tag.",
        "For indexing a cache directory without opening every file.",
    ),

    # -- view --------------------------------------------------------------
    Export(
        "ViewPredictor", ".geo.view", "view",
        "What a camera at a known pose should see of the map: corners and skyline.",
        "The prediction half of any image-to-map match. sweep_headings() does the "
        "pose-independent work once for a whole circle.",
    ),
    Export(
        "PredictedView", ".geo.view", "view",
        "The map's answer for one pose: visible edges and the skyline behind them.",
        "Feed visible_edges to a heading match, or draw it over a frame.",
    ),
    Export(
        "VerticalEdge", ".geo.view", "view",
        "A building corner as the camera would see it, with its bearing and range.",
        "pixel_column() places it in a frame. Check occluded before using it: a "
        "nearer footprint spanning the same bearing hides it whatever the heights.",
    ),
    Export(
        "SkylinePoint", ".geo.view", "view",
        "The highest terrain along one bearing, and how far up it sits.",
        "What remains where there are no buildings, which is about half the area.",
    ),
    Export(
        "bearing_and_range", ".geo.view", "view",
        "Bearing and range from one geographic point to another.",
        "The tangent-plane conversion the rest of the view code runs on.",
    ),
    Export(
        "relative_bearing", ".geo.view", "view",
        "An absolute bearing expressed relative to where the camera points.",
        "Wraps correctly across north; do not subtract bearings by hand.",
    ),

    # -- calibration -------------------------------------------------------
    Export(
        "solve_imu_camera", ".calibration.imu", "calibration",
        "The fixed rotation from IMU to camera, solved from gravity in several poses.",
        "Perception's section 3: without it, 'gravity down' and yaw rate do not sit "
        "on the axes the model works in. Wahba's problem, closed form by SVD.",
    ),
    Export(
        "ImuCameraSolution", ".calibration.imu", "calibration",
        "That rotation, with its residuals and how well its weakest axis is pinned.",
        "Read observability before trusting it: two gravity directions 25 degrees "
        "apart pin the weak axis about ten times less than 90 degrees apart.",
    ),
    Export(
        "ImuCameraConstraint", ".calibration.imu", "calibration",
        "The same solve as a pipeline constraint, validating before it solves.",
        "Use this rather than the bare solver when the answer feeds a guarded start.",
    ),
    Export(
        "RigOrientation", ".calibration.imu", "calibration",
        "One static pose: what the IMU read, and where the camera was pointing.",
        "Both directions point UP, matching what the accelerometer reports at rest. "
        "Flipping one sign and not the other gives a 180-degree error with clean "
        "residuals, which nothing downstream can catch.",
    ),
    Export(
        "ImuCalibrationFailed", ".calibration.imu", "calibration",
        "Raised when the rotation cannot be solved from what was provided.",
        "Names the cause: too few static poses, or tilts too alike to fix the "
        "third axis.",
    ),
    Export(
        "CalibrationDependencyMissing", ".calibration.imu", "calibration",
        "Raised when a package the solver needs is not installed.",
        "A subclass of ImuCalibrationFailed, so catching the parent still works. "
        "Branch on it to tell a bad install from a bad capture session, rather "
        "than sending someone back to the rig to repeat a session that was fine.",
    ),
    Export(
        "tilt_separation_deg", ".calibration.imu", "calibration",
        "The largest angle between any two gravity directions in a set of poses.",
        "Check it before a capture session ends, while the rig is still out.",
    ),
    Export(
        "SurveyedTarget", ".calibration.targets", "calibration",
        "A target whose position is known better than either sensor measures it.",
        "Known to 5 mm, so it acts as truth rather than as another unknown.",
    ),
    Export(
        "SurveyedTargetConstraint", ".calibration.targets", "calibration",
        "A set of surveyed targets as a calibration constraint.",
        "Validates that the set can constrain a solve before one is attempted.",
    ),

    # -- navigation --------------------------------------------------------
    Export(
        "DeadReckoner", ".nav.propagation", "navigation",
        "How far position drifts by a given horizon, decomposed by cause.",
        "budget_at(seconds) for the breakdown, horizon_for(metres) for how long "
        "a fix survives a limit. Answers how often a correction is needed.",
    ),
    Export(
        "DriftBudget", ".nav.propagation", "navigation",
        "Position error at one horizon, split by source rather than summed.",
        "Read dominant to know what to fix. Read rests_on_assumption before "
        "quoting the total: an unmeasured constant makes it a prediction.",
    ),
    Export(
        "DriftTerm", ".nav.propagation", "navigation",
        "One contributor to that error, in metres, with what produced it.",
        "The named terms are what make a budget actionable.",
    ),
    Export(
        "ImuErrorModel", ".nav.propagation", "navigation",
        "The five sensor constants that decide how fast an unaided fix decays.",
        "heading_sigma_deg(moving_s) reproduces the rig's own yaw_sigma formula, "
        "so the two agree by construction rather than by coincidence.",
    ),
    Export(
        "NoiseTerm", ".nav.propagation", "navigation",
        "One sensor constant, carrying whether it was measured or assumed.",
        "A budget is only as good as its weakest constant, and provenance is what "
        "keeps that visible once every number is a bare float.",
    ),
    Export(
        "RIG_ERROR_MODEL", ".nav.propagation", "navigation",
        "The shipped default model, with its unmeasured constants marked.",
        "A starting point, not a description of your rig. Replace it with "
        "identify() on a real recording.",
    ),
    Export(
        "NavState", ".nav.propagation", "navigation",
        "A pose on the local tangent plane with its 4x4 covariance.",
        "Carries the gyro bias as a state, not as heading process noise: folding "
        "it in gives the right heading sigma and understates cross-track by "
        "sqrt(2/3), which is optimism that survives review.",
    ),
    Export(
        "SpeedAiding", ".nav.propagation", "navigation",
        "Speed from an external source, and how well it is known.",
        "Radar ego-velocity is the intended supplier. Its presence changes the "
        "regime, not just the constant.",
    ),
    Export(
        "advance", ".nav.propagation", "navigation",
        "Propagate one state forward a single step, covariance included.",
        "is_static freezes heading, its uncertainty and the speed error, but not "
        "position uncertainty already accrued.",
    ),
    Export(
        "propagate", ".nav.propagation", "navigation",
        "Run a state through a sequence of (dt, speed, yaw rate) samples.",
        "The bulk form of advance, for replaying a recorded run.",
    ),
    Export(
        "PropagationError", ".nav.propagation", "navigation",
        "Raised when a budget is asked for something it cannot answer.",
        "Negative times, non-positive limits, negative speeds.",
    ),
    Export(
        "STANDARD_GRAVITY", ".nav.propagation", "navigation",
        "Standard gravity in m/s^2.",
        "The constant that turns a tilt error into an acceleration error, and so "
        "sets the unaided horizon.",
    ),
    Export(
        "WallMatcher", ".nav.walls", "navigation",
        "Solves position and heading by matching radar wall returns to building footprints.",
        "The map-relative fix: a prior (GNSS, manual pin, dead reckoning) plus the static "
        "returns from radar_detections_all(). Read `accepted` and `ambiguous`, not just the pose.",
    ),
    Export(
        "WallReturn", ".nav.walls", "navigation",
        "One static radar return: range and azimuth (positive = right of boresight).",
        "Build these from radar_detections_all() records with static_returns().",
    ),
    Export(
        "PosePrior", ".nav.walls", "navigation",
        "Where the rig believes it is before the walls are consulted, with sigmas.",
        "Bounds the search; a rival street outside 3 sigma cannot win. Sigmas must be honest.",
    ),
    Export(
        "WallFix", ".nav.walls", "navigation",
        "The pose that best explains the returns, its sigmas, and whether to trust it.",
        "`ambiguous` names the axis a single wall cannot pin; `on_boundary` says the prior lied.",
    ),
    Export(
        "static_returns", ".nav.walls", "navigation",
        "Filters radar_detections_all() records to wall returns: static, 0.5-40 m.",
        "Feed the result to WallMatcher or PoseObservations.",
    ),
    Export(
        "PoseObservations", ".stages.pose", "map",
        "The pose stage's input: a PosePrior and the static wall returns.",
        "Pass to MapInitializer(pose_observations=...) to make the pose_init stage run.",
    ),
    Export(
        "InitialPose", ".stages.pose", "map",
        "A map-relative pose with the sigmas that qualify it, as the pose stage publishes it.",
        "Read from report.stage('pose_init').data['pose'].",
    ),
    Export(
        "HeadingMatcher", ".nav.heading", "navigation",
        "Recovers heading by matching observed edge bearings against the map.",
        "The only correction heading has. A prior narrows the search and, more "
        "importantly, excludes distant look-alike peaks.",
    ),
    Export(
        "HeadingFix", ".nav.heading", "navigation",
        "A heading recovered from one frame, with the caveats that qualify it.",
        "Check accepted, not just heading_deg. An ambiguous fix on a repeating "
        "street is worse than none, because everything downstream multiplies "
        "heading by range.",
    ),
    Export(
        "HeadingCandidate", ".nav.heading", "navigation",
        "One heading the search evaluated, and how well it explained the frame.",
        "HeadingFix.peaks holds these; a caller with its own prior can pick from "
        "them instead of taking the fix's choice.",
    ),
    Export(
        "HeadingMatchError", ".nav.heading", "navigation",
        "Raised when a heading cannot be recovered from what was given.",
        "An empty prediction is a real state with no answer, not a zero heading.",
    ),
    Export(
        "bearings_from_columns", ".nav.heading", "navigation",
        "Convert pixel columns to bearings using measured intrinsics.",
        "Uses fx and cx, not a field of view: the two agree only for an ideal "
        "lens. Columns must already have distortion removed -- this rig's k1 of "
        "-0.366 moves a corner pixel about ten columns.",
    ),
    Export(
        "bearings_from_view", ".nav.heading", "navigation",
        "Absolute bearings of the visible corners in a PredictedView.",
        "The bridge from the view side to the matcher.",
    ),
    Export(
        "angular_difference_deg", ".nav.heading", "navigation",
        "Signed difference between two bearings, wrapped to [-180, 180).",
        "Bearing wrap is where a matcher quietly breaks; use this, not subtraction.",
    ),
    Export(
        "identify", ".nav.identification", "navigation",
        "Fit sensor constants from a stationary recording off the rig.",
        "Turns a drift budget from a prediction into a measurement. What a still "
        "rig cannot settle stays marked assumed.",
    ),
    Export(
        "load_recording", ".nav.identification", "navigation",
        "Read an imu.csv and holds.json pair written by the rig's hold tool.",
        "The input to identify().",
    ),
    Export(
        "Recording", ".nav.identification", "navigation",
        "A loaded recording: its samples, its stationary windows, its duration.",
        "longest_hold() is what the random walk is read from.",
    ),
    Export(
        "Hold", ".nav.identification", "navigation",
        "One stationary window inside a recording.",
        "Short holds bound what can be identified; check the duration.",
    ),
    Export(
        "RecordingError", ".nav.identification", "navigation",
        "Raised when a recording cannot be read or is too short to fit from.",
        "Better than a constant fitted from three seconds of data.",
    ),
    Export(
        "allan_deviation", ".nav.identification", "navigation",
        "Allan deviation of a sample series across a range of averaging times.",
        "The curve every inertial constant is read off. Its slope names the noise.",
    ),
    Export(
        "angle_random_walk_dps_sqrt_s", ".nav.identification", "navigation",
        "Angle random walk read off that curve at one second.",
        "Needs only seconds of stationary data, unlike bias instability.",
    ),
    Export(
        "identification_checks", ".nav.identification", "navigation",
        "Whether the recording actually supports the constants fitted from it.",
        "Fails loudly when a recording is too short to reach the Allan floor, so "
        "an upper bound is not quoted as a measurement.",
    ),
)


#: Which optional dependency group an area needs, for areas that need one.
#: health and navigation are absent on purpose: they are pure arithmetic and
#: install with nothing.
AREA_EXTRA: Dict[str, str] = {
    "geo": "geo",
    "view": "geo",
    "map": "geo",
    "calibration": "calibration",
}


def extra_for(name: str) -> Optional[str]:
    """The optional dependency group a public name needs, or None if it needs none."""
    for export in PUBLIC_API:
        if export.name == name:
            return AREA_EXTRA.get(export.area)
    return None


def exports_by_area() -> Dict[str, List[Export]]:
    """Every export grouped under its area, in the order AREAS declares."""
    grouped: Dict[str, List[Export]] = {area: [] for area, _ in AREAS}
    for export in PUBLIC_API:
        grouped[export.area].append(export)
    return grouped


def describe(area: Optional[str] = None) -> str:
    """The catalogue as readable text, whole or for one area.

    Written for a person deciding what to import, so it leads with what a thing
    is rather than with where it lives.
    """
    grouped = exports_by_area()
    if area is not None:
        if area not in grouped:
            known = ", ".join(name for name, _ in AREAS)
            raise KeyError(f"unknown area {area!r}; known areas are {known}")
        selected = [(area, dict(AREAS)[area])]
    else:
        selected = list(AREAS)

    lines: List[str] = []
    for name, blurb in selected:
        lines.append(f"\n{name.upper()} -- {blurb}")
        lines.append("-" * 78)
        for export in grouped[name]:
            lines.append(f"{export.name}")
            lines.append(f"    {export.summary}")
            lines.append(f"    use: {export.use}")
            lines.append(f"    from: mapinit{export.module}")
    return "\n".join(lines).strip()

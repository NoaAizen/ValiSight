"""Compatibility imports for the student models and training helpers.

Architectures live in :mod:`perception.models`; datasets and optimization live
in :mod:`perception.learning`. Existing scripts and saved Python object paths
can continue importing their original names from this module.
"""
from perception.models.features import (
    EPS,
    Derived2DChannels,
    DerivedRangeProfileChannels,
    PointFeatureBuilder,
)
from perception.models.students import (
    Spatial2DEncoder,
    RangeProfileEncoder,
    PointSetEncoder,
    DetectionHead,
    ThermalStudent,
    RadarStudent,
)
from perception.learning.data import (
    augment_thermal_seq,
    augment_radar_points,
    StudentDataset,
    build_loaders,
    radar_stream_stats,
)
from perception.learning.losses import (
    detection_loss,
)
from perception.learning.engine import (
    move_batch,
    evaluate_model,
    train_model,
)

__all__ = [
    "EPS",
    "Derived2DChannels",
    "DerivedRangeProfileChannels",
    "PointFeatureBuilder",
    "Spatial2DEncoder",
    "RangeProfileEncoder",
    "PointSetEncoder",
    "DetectionHead",
    "ThermalStudent",
    "RadarStudent",
    "augment_thermal_seq",
    "augment_radar_points",
    "StudentDataset",
    "build_loaders",
    "radar_stream_stats",
    "detection_loss",
    "move_batch",
    "evaluate_model",
    "train_model",
]

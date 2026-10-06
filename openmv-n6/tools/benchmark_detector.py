#!/usr/bin/env python3
"""Measure the full detector call on saved images, without opening sensors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import cv2
import numpy as np

from detector_models import MODEL_FILES
from detect import COCO
from trt_detect import TrtDetector


def read_image(path):
    if path.suffix == '.raw':
        data = np.fromfile(path, dtype=np.uint8)
        if data.size != 640 * 400:
            raise ValueError(f'{path}: expected a 640x400 visible luma frame')
        return data.reshape(400, 640)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(MODEL_FILES), default='yolo26n')
    parser.add_argument('--engine')
    parser.add_argument('--images', nargs='+', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error('warmup must be nonnegative and iterations must be positive')
    images = [read_image(path) for path in args.images]
    detector = TrtDetector(model=args.model, engine=args.engine, names=COCO)
    try:
        for i in range(args.warmup):
            detector(images[i % len(images)])
        elapsed = []
        for i in range(args.iterations):
            start = time.perf_counter()
            detector(images[i % len(images)])
            elapsed.append((time.perf_counter() - start) * 1000)
        samples = [{'image': str(path), 'shape': list(img.shape),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'detections': detector(img)} for path, img in zip(args.images, images)]
        import tensorrt
        report = {
            'model': args.model, 'engine': detector.engine_path,
            'engine_sha256': hashlib.sha256(Path(detector.engine_path).read_bytes()).hexdigest(),
            'tensorrt': tensorrt.__version__, 'architecture': platform.machine(),
            'input_shape': list(detector._in_shape), 'output_shape': list(detector._out_shape),
            'iterations': args.iterations, 'warmup': args.warmup,
            'latency_ms': {'mean': float(np.mean(elapsed)), 'p50': float(np.median(elapsed)),
                           'p95': float(np.percentile(elapsed, 95)), 'max': float(max(elapsed))},
            'detector_calls_per_second': 1000 / float(np.mean(elapsed)),
            'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            'scope': 'grayscale, preprocess + copies + synchronized inference + decode; '
                     'excludes image loading, sensors, fusion and rendering; not an accuracy evaluation',
            'samples': samples,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))
    finally:
        detector.close()


if __name__ == '__main__':
    main()

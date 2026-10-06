#!/usr/bin/env python3
"""Compare decoded ONNX CPU predictions with TensorRT FP16 on one saved image."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_detector import read_image
from trt_detect import TrtDetector, validate_io_shapes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--onnx', type=Path, help='Create a reference with ONNX Runtime CPU')
    modes.add_argument('--model', help='Compare the registered TensorRT model against the reference')
    parser.add_argument('--image', type=Path)
    parser.add_argument('--engine', help='Override the registered TensorRT engine')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.onnx:
        if not args.image:
            parser.error('--onnx requires --image')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(args.onnx), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        inp, out = session.get_inputs()[0], session.get_outputs()[0]
        validate_io_shapes(inp.shape, out.shape)
        # Reuse production preprocessing, with no CUDA allocation.
        prep = object.__new__(TrtDetector)
        prep.net_h, prep.net_w = inp.shape[2:]
        prep.h_in = np.empty(inp.shape, dtype=np.float32)
        prep._geom, prep._pad_val = None, 114 / 255
        image = read_image(args.image)
        prep.letterbox(image)
        output = session.run([out.name], {inp.name: prep.h_in})[0]
        np.savez_compressed(args.reference, image=image, input=prep.h_in, output=output)
        print(f'ONNX CPU reference saved: {args.reference}')
        return

    from scipy.optimize import linear_sum_assignment
    reference = np.load(args.reference, allow_pickle=False)
    detector = TrtDetector(model=args.model, engine=args.engine)
    try:
        detector(reference['image'])
        np.testing.assert_allclose(detector.h_in, reference['input'], atol=1e-3, rtol=1e-3)
        expected, actual = reference['output'][0], detector.h_out[0]
        # Compare confident detections; low-score top-k rows are unstable in FP16.
        expected = expected[expected[:, 4] >= .5]
        actual = actual[actual[:, 4] >= .5]
        if not len(expected) or len(expected) != len(actual):
            raise RuntimeError(f'Need equal, nonempty confident detections: '
                               f'ONNX={len(expected)}, TensorRT={len(actual)}')
        a, b = expected[:, None, :4], actual[None, :, :4]
        wh = np.maximum(0, np.minimum(a[..., 2:], b[..., 2:]) - np.maximum(a[..., :2], b[..., :2]))
        intersection = wh.prod(axis=-1)
        union = ((a[..., 2:] - a[..., :2]).prod(axis=-1)
                 + (b[..., 2:] - b[..., :2]).prod(axis=-1) - intersection)
        iou = intersection / np.maximum(union, 1e-9)
        cost = 1 - iou + (expected[:, None, 5] != actual[None, :, 5]) * 100
        rows, cols = linear_sum_assignment(cost)
        min_iou = float(iou[rows, cols].min())
        score_error = float(np.abs(expected[rows, 4] - actual[cols, 4]).max())
        passed = (np.array_equal(expected[rows, 5], actual[cols, 5])
                  and min_iou >= .95 and score_error <= .03)
        report = {'model': args.model, 'passed': bool(passed),
                  'confident_detections': len(rows), 'minimum_box_iou': min_iou,
                  'maximum_score_error': score_error,
                  'scope': 'numerical export parity on one grayscale image; not detection accuracy'}
        if args.out:
            args.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        detector.close()


if __name__ == '__main__':
    main()

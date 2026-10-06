"""Thermal vehicle baseline: smoke test, training, resumable checkpoints.

Run in the pinned Jetson container. This does not install or activate a live
model. Animal classes are absent because the supplied data cannot support them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
os.environ.setdefault('TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT', '64')

NAMES = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']


def subset(root, split, count, out):
    """Small deterministic sample with coverage for rare vehicle categories."""
    images = sorted((root / 'images' / split).glob('*.png'))
    random.Random(20260908).shuffle(images)
    selected, counts = [], [0] * len(NAMES)
    for image in images:
        labels = root / 'labels' / split / (image.stem + '.txt')
        classes = {int(row.split()[0]) for row in labels.read_text().splitlines() if row.strip()}
        if any(counts[k] < 4 for k in classes):
            selected.append(image)
            for k in classes:
                counts[k] += 1
        if all(c >= 4 for c in counts):
            break
    selected += [p for p in images if p not in selected][:max(0, count - len(selected))]
    out.write_text('\n'.join(map(str, selected)) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO, settings
    import yaml

    # Training and checkpoints stay local; no external experiment reporting.
    settings.update({'sync': False, 'wandb': False, 'mlflow': False,
                     'comet': False, 'clearml': False, 'neptune': False})
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; refusing an accidental CPU training run')
    print('CUDA free/total bytes:', torch.cuda.mem_get_info(), flush=True)
    config = yaml.safe_load(args.data.read_text())
    if config['names'] != NAMES:
        raise ValueError('Unexpected class ordering in dataset')
    args.out.mkdir(parents=True, exist_ok=True)
    status = args.out / ('smoke-status.json' if args.smoke else 'training-status.json')
    run = 'smoke' if args.smoke else 'baseline'
    data = args.data
    if args.smoke:
        root = Path(config['path'])
        for split in ('train', 'val'):
            subset(root, split, 64, args.out / (split + '-smoke.txt'))
            config[split] = str(args.out / (split + '-smoke.txt'))
        data = args.out / 'smoke-data.yaml'
        data.write_text(yaml.safe_dump(config))

    provenance = {'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
                  'gpu': torch.cuda.get_device_name(0), 'classes': NAMES,
                  'data': str(data), 'image_size': 320, 'batch': args.batch,
                  'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  'scope': 'FLIR thermal vehicles/person baseline; no animal capability; '
                           'not yet validated on Lepton recordings', 'started_at': time.time()}

    def write_status(**values):
        tmp = status.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(provenance, updated_at=time.time(), **values), indent=2) + '\n')
        tmp.replace(status)

    def epoch_saved(trainer):
        write_status(state='training', epoch_completed=trainer.epoch + 1,
                     epochs=trainer.epochs, checkpoint=str(trainer.last),
                     metrics={k: float(v) for k, v in trainer.metrics.items()})

    write_status(state='starting')
    try:
        # Load CUDA/cuBLAS before training allocates activation buffers.
        # Tiny elementwise-only probes do not exercise the GEMM backend.
        from ultralytics.utils.torch_utils import init_seeds
        init_seeds(20260908, deterministic=True)
        probe = torch.ones((32, 32), device='cuda', requires_grad=True)
        (probe @ probe).sum().backward()
        torch.cuda.synchronize()
        del probe
        model = YOLO(str(args.resume or args.weights))
        model.to('cuda:0')
        model.add_callback('on_model_save', epoch_saved)
        if args.resume:
            model.train(resume=True, device=0, workers=0)
        else:
            model.train(data=str(data), epochs=1 if args.smoke else args.epochs,
                        imgsz=320, batch=args.batch, device=0, workers=0,
                        cache=False, amp=False, pretrained=True, seed=20260908,
                        deterministic=True, optimizer='AdamW', lr0=0.001,
                        patience=8, hsv_h=0, hsv_s=0, hsv_v=0.15,
                        mosaic=0.5, close_mosaic=0 if args.smoke else 5,
                        mixup=0, fliplr=0.5, flipud=0, plots=False,
                        project=str(args.out), name=run, exist_ok=False,
                        save=True, save_period=1, verbose=False)
        best = Path(model.trainer.best)
        if not best.is_file():
            raise RuntimeError('Training finished without a best checkpoint')
        write_status(state='completed', best=str(best),
                     epoch_completed=model.trainer.epoch + 1,
                     metrics={k: float(v) for k, v in model.trainer.metrics.items()})
    except Exception as exc:
        write_status(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()

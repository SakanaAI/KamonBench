from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

SIZES = [2500, 5000, 10000]
FAMILIES = ['vit', 'vgg_mask', 'vgg_no_mask']
LABELS = ['jp', 'en', 'program']


@dataclass(frozen=True)
class Job:
    size: int
    family: str
    label: str

    @property
    def name(self) -> str:
        return f'{self.family}_{self.label}'


def run_checked(cmd: list[str], *, cwd: Path, log: Path | None = None, env: dict[str, str] | None = None) -> None:
    if log is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as f:
        subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def croissant_for(croissant_root: Path, job: Job) -> Path:
    suffix = '_program' if job.label == 'program' else ''
    return croissant_root / f'train_{job.size}' / f'kamon_croissant{suffix}.json'


def train_args(job: Job, *, croissant: Path, checkpoint_dir: Path, output_dir: Path) -> list[str]:
    args = [
        'train',
        '--croissant-path',
        str(croissant),
        '--checkpoint-dir',
        str(checkpoint_dir),
        '--output-dir',
        str(output_dir),
        '--device',
        'cuda:0',
    ]
    if job.label == 'en':
        args.append('--use-translation')

    if job.family == 'vit':
        return args + [
            '--model',
            'vit_decoder',
            '--batch-size',
            '64',
            '--num-epochs',
            '50',
            '--checkpoint-steps',
            '2000',
            '--early-stop-patience',
            '8',
            '--num-train-augmentations',
            '3',
            '--learning-rate',
            '0.00015',
            '--vit-backbone-learning-rate',
            '0.00001',
            '--weight-decay',
            '0.0001',
            '--label-smoothing',
            '0.02',
        ]
    if job.family == 'vgg_mask':
        return args + [
            '--model',
            'vgg',
            '--ngram-length',
            '1',
            '--batch-size',
            '64',
            '--num-epochs',
            '200',
            '--checkpoint-steps',
            '1250',
            '--early-stop-patience',
            '10',
            '--num-train-augmentations',
            '5',
            '--learning-rate',
            '0.0001',
            '--weight-decay',
            '0.0001',
            '--label-smoothing',
            '0.02',
            '--eval-composites-only',
        ]
    if job.family == 'vgg_no_mask':
        return args + [
            '--model',
            'vgg',
            '--no-masks',
            '--ngram-length',
            '4',
            '--batch-size',
            '64',
            '--num-epochs',
            '200',
            '--checkpoint-steps',
            '1250',
            '--early-stop-patience',
            '10',
            '--num-train-augmentations',
            '5',
            '--learning-rate',
            '0.0001',
            '--weight-decay',
            '0.0001',
            '--label-smoothing',
            '0.02',
            '--eval-composites-only',
        ]
    raise ValueError(job.family)


def best_checkpoint(checkpoint_dir: Path) -> Path | None:
    matches = sorted(checkpoint_dir.glob('checkpoint_best_*.pt'))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(f'Multiple best checkpoints in {checkpoint_dir}')
    return matches[0]


def job_paths(sweep_root: Path, job: Job) -> tuple[Path, Path, Path]:
    size_root = sweep_root / f'train_{job.size}'
    return size_root / 'checkpoints' / job.name, size_root / 'runs' / job.name, size_root / 'logs'


def training_command(project_root: Path, code_root: Path, sweep_root: Path, croissant_root: Path, job: Job):
    checkpoint_dir, output_dir, log_dir = job_paths(sweep_root, job)
    if (output_dir / 'step_final' / 'predictions.jsonl').exists() and best_checkpoint(checkpoint_dir):
        return None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(code_root / '.venv' / 'bin' / 'python'), '-m', 'kamonbench'] + train_args(
        job, croissant=croissant_for(croissant_root, job), checkpoint_dir=checkpoint_dir, output_dir=output_dir
    )
    return cmd, log_dir / f'train_{job.name}.log'


def inference_command(code_root: Path, sweep_root: Path, croissant_root: Path, job: Job):
    checkpoint_dir, output_dir, log_dir = job_paths(sweep_root, job)
    test_jsonl = output_dir / 'test.jsonl'
    if test_jsonl.exists():
        return None
    best = best_checkpoint(checkpoint_dir)
    if best is None:
        raise RuntimeError(f'Missing best checkpoint for {job}')
    args = [
        'infer',
        '--checkpoint-path',
        str(best),
        '--output-file',
        str(test_jsonl),
        '--croissant-path',
        str(croissant_for(croissant_root, job)),
        '--batch-size',
        '64',
        '--device',
        'cuda:0',
    ]
    if job.label == 'en':
        args.append('--use-translation')
    cmd = [str(code_root / '.venv' / 'bin' / 'python'), '-m', 'kamonbench'] + args
    return cmd, log_dir / f'infer_{job.name}.log'


def run_parallel(*, phase: str, jobs: list[Job], gpus: list[str], cwd: Path, build_command) -> None:
    pending = deque(jobs)
    free_gpus = deque(gpus)
    running: list[tuple[subprocess.Popen, str, Job, Path]] = []

    while pending or running:
        while pending and free_gpus:
            job = pending.popleft()
            built = build_command(job)
            if built is None:
                print(f'==> skip {phase} train_{job.size} {job.name}', flush=True)
                continue
            cmd, log = built
            gpu = free_gpus.popleft()
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            env['PYTHONUNBUFFERED'] = '1'
            print(f'==> {phase} train_{job.size} {job.name} on GPU {gpu}', flush=True)
            f = log.open('w', encoding='utf-8')
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
            proc._kamon_log_file = f  # type: ignore[attr-defined]
            running.append((proc, gpu, job, log))

        time.sleep(15)
        still_running = []
        for proc, gpu, job, log in running:
            status = proc.poll()
            if status is None:
                still_running.append((proc, gpu, job, log))
                continue
            proc._kamon_log_file.close()  # type: ignore[attr-defined]
            free_gpus.append(gpu)
            if status != 0:
                raise RuntimeError(f'{phase} failed for train_{job.size} {job.name}; see {log}')
            print(f'==> done {phase} train_{job.size} {job.name}', flush=True)
        running = still_running


def summarize_job(project_root: Path, code_root: Path, sweep_root: Path, croissant_root: Path, job: Job) -> None:
    _, output_dir, log_dir = job_paths(sweep_root, job)
    test_jsonl = output_dir / 'test.jsonl'
    summary_json = output_dir / 'test_summary_slices.json'
    factor_json = output_dir / 'test_factors.json'
    python = str(code_root / '.venv' / 'bin' / 'python')
    if not summary_json.exists():
        run_checked(
            [
                python,
                '-m',
                'kamonbench',
                'summarize-predictions',
                str(test_jsonl),
                '--croissant-path',
                str(croissant_for(croissant_root, job)),
                '--split',
                'test',
                '--include-slices',
                '--bootstrap-samples',
                '1000',
                '--output',
                str(summary_json),
            ],
            cwd=project_root,
            log=log_dir / f'summary_{job.name}.log',
        )
    if job.label == 'program' and not factor_json.exists():
        run_checked(
            [
                python,
                '-m',
                'kamonbench',
                'evaluate-program-factors',
                str(test_jsonl),
                '--output',
                str(factor_json),
                '--markdown-output',
                str(output_dir / 'test_factors.md'),
            ],
            cwd=project_root,
            log=log_dir / f'factors_{job.name}.log',
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', default=os.environ.get('KAMON_SWEEP_GPUS', '0 1 2').split())
    parser.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    code_root = script_dir.parent
    project_root = code_root.parent.parent
    python = code_root / '.venv' / 'bin' / 'python'
    sweep_root = project_root / 'baselines' / 'train_size_sweeps'
    croissant_root = sweep_root / 'croissants'
    jobs = [Job(size, family, label) for size in args.sizes for family in FAMILIES for label in LABELS]

    run_checked(
        [
            str(python),
            str(script_dir / 'make_train_size_croissants.py'),
            '--source-croissant',
            str(project_root / 'dataset01' / 'kamon_croissant.json'),
            '--program-croissant',
            str(project_root / 'etc' / 'source' / 'kamon_croissant_program.json'),
            '--output-dir',
            str(croissant_root),
            '--sizes',
            *[str(size) for size in args.sizes],
        ],
        cwd=project_root,
    )
    run_parallel(
        phase='train',
        jobs=jobs,
        gpus=args.gpus,
        cwd=project_root,
        build_command=lambda job: training_command(project_root, code_root, sweep_root, croissant_root, job),
    )
    run_parallel(
        phase='infer',
        jobs=jobs,
        gpus=args.gpus,
        cwd=project_root,
        build_command=lambda job: inference_command(code_root, sweep_root, croissant_root, job),
    )
    for job in jobs:
        print(f'==> summarize train_{job.size} {job.name}', flush=True)
        summarize_job(project_root, code_root, sweep_root, croissant_root, job)
    run_checked(
        [
            str(python),
            str(script_dir / 'write_train_size_tables.py'),
            '--sweep-root',
            str(sweep_root),
            '--latex-root',
            str(project_root / 'latex' / 'KamonBench'),
            '--sizes',
            *[str(size) for size in args.sizes],
        ],
        cwd=project_root,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

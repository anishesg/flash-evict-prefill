"""Run the authorized performance/quality experiments sequentially on one GPU.

Each completed experiment is committed and pushed. Prediction-level resume lives
in evaluate_longbench.py; benchmark resume lives in benchmark_model.py. This
driver never terminates other users' processes or changes experiment parameters.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_status(phase, **extra):
    path = Path('results/campaign_status.json')
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(phase=phase, updated_unix=time.time(), driver_pid=os.getpid(), **extra), indent=2)+'\n')
    tmp.replace(path)
    print(json.dumps(dict(phase=phase, **extra)), flush=True)


def run(command, log):
    with Path(log).open('a') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        write_status(command[1], child_pid=process.pid, log=log)
        return process.wait()


def push(message, paths):
    subprocess.run(['git', 'add', '--', *paths], check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode:
        subprocess.run(['git', 'commit', '-m', 'flash-evict: '+message], check=True)
        subprocess.run(['git', 'push', 'origin', 'master'], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    os.environ['PYTHONPATH'] = '.:/home/qcb/kv-distill'
    python = sys.executable
    if args.wait_pid:
        write_status('waiting_for_running_model_benchmark', child_pid=args.wait_pid)
        while True:
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(5)
    common = [python, 'scripts/benchmark_model.py', '--policies', 'results/priority_policies.json',
              '--names', 'd01_r0', 'd0.1_w128_r0', 'd0.1_w128_r64', '--require-idle']
    configurations=[('model_benchmark', [], None),
                    ('model_benchmark_chunked', ['--mlp-chunk-size', '1024'], None),
                    ('model_benchmark_expandable', ['--mlp-chunk-size', '1024'], 'expandable_segments:True')]
    for name, extra, allocator in configurations:
        output = f'results/{name}.json'
        while not (Path(output).exists() and json.loads(Path(output).read_text()).get('complete')):
            if allocator:
                os.environ['PYTORCH_CUDA_ALLOC_CONF']=allocator
            code = run(common+extra+['--output', output], f'results/{name}.log')
            if allocator:
                os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
                if Path(output).exists():
                    report=json.loads(Path(output).read_text())
                    report.update(allocator_configuration=allocator,
                                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                    Path(output).write_text(json.dumps(report,indent=2)+'\n')
            if code:
                # Retry only explicitly identified GPU contention; other errors
                # require investigation instead of silently altering the protocol.
                tail = Path(f'results/{name}.log').read_text()[-4000:]
                if 'Contention observed' not in tail and 'Other CUDA processes' not in tail:
                    raise RuntimeError(f'{name} failed with exit code {code}')
                time.sleep(10)
        subprocess.run([python, 'scripts/report_followup.py', '--partial'], check=True)
        push(f'complete {name} across 28 Qwen layers',
             [output, 'results/FOLLOWUP.md', 'results/FOLLOWUP.json', 'results/campaign_status.json'])
    smoke = Path('results/longbench_smoke/complete.json')
    if not smoke.exists():
        command = [python, 'scripts/evaluate_longbench.py', '--output', 'results/longbench_smoke',
                   '--tasks', 'narrativeqa', 'qasper', 'multifieldqa_zh', '--limit', '1', '--require-idle', '--push']
        if run(command, 'results/longbench_smoke.log'):
            raise RuntimeError('LongBench smoke evaluation failed')
    command = [python, 'scripts/evaluate_longbench.py', '--require-idle', '--push']
    if run(command, 'results/longbench_full.log'):
        raise RuntimeError('Full LongBench evaluation stopped; inspect its failure record and resume')
    summary = json.loads(Path('results/longbench_full/SUMMARY.json').read_text())
    if not summary['complete']:
        raise RuntimeError('Full LongBench coverage audit failed')
    write_status('complete')
    push('complete requested performance table and full LongBench campaign', ['results/campaign_status.json'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write_status('stopped_with_error', error=repr(error))
        raise

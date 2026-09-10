import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def fixture_run(root, *, missing=False, duplicate=False):
    methods = ['d01_r0_256', 'snap_256']
    tasks = ['narrativeqa', 'qasper']
    (root/'protocol.json').write_text(json.dumps(dict(tasks=tasks, methods=[dict(name=m) for m in methods],
                                                    limit=None, mlp_chunk_size=1024)))
    (root/'data_manifest.json').write_text(json.dumps(dict(context_limit=32768,
        tasks={'narrativeqa': dict(examples=3), 'qasper': dict(examples=1)})))
    for method in methods:
        folder = root/'pred'/method
        folder.mkdir(parents=True)
        for task, values in [('narrativeqa', [80, 20, 20]), ('qasper', [60])]:
            rows = [dict(id=f'{task}:{i}', score=value-(5 if method == 'snap_256' else 0),
                         used_in_tuning=i == 0 and task == 'narrativeqa') for i, value in enumerate(values)]
            if missing and method == 'snap_256' and task == 'narrativeqa':
                rows.pop()
            if duplicate and method == 'snap_256' and task == 'narrativeqa':
                rows.append(rows[0])
            (folder/f'{task}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def report(root):
    return subprocess.run([sys.executable, str(ROOT/'scripts/report_longbench.py'), '--output', str(root)],
                          capture_output=True, text=True)


def test_task_macro_and_paired_interval(tmp_path):
    fixture_run(tmp_path)
    result = report(tmp_path)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path/'SUMMARY.json').read_text())
    assert summary['complete']
    assert summary['methods']['d01_r0_256']['macro_all'] == 50
    assert summary['methods']['d01_r0_256']['macro_excluding_tuned'] == 40
    paired = summary['comparisons']['d01_r0_256 minus snap_256']
    assert paired['macro_difference'] == 5
    assert paired['paired_stratified_bootstrap_95ci'] == [5, 5]


def test_partial_run_never_emits_full_score(tmp_path):
    fixture_run(tmp_path, missing=True)
    assert report(tmp_path).returncode == 0
    summary = json.loads((tmp_path/'SUMMARY.json').read_text())
    assert not summary['complete']
    assert summary['methods']['snap_256']['macro_all'] is None
    assert summary['methods']['snap_256']['completed_task_scores'] == {'qasper': 55}
    assert not summary['comparisons']


def test_duplicate_records_fail_coverage_audit(tmp_path):
    fixture_run(tmp_path, duplicate=True)
    result = report(tmp_path)
    assert result.returncode != 0
    assert 'Duplicate records' in result.stderr


def test_changed_prompt_hash_fails_even_with_full_counts(tmp_path):
    fixture_run(tmp_path)
    path = tmp_path/'data_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['examples'] = [dict(id=f'{task}:{i}', task=task, input_ids_sha256='frozen')
                            for task, count in [('narrativeqa', 3), ('qasper', 1)] for i in range(count)]
    path.write_text(json.dumps(manifest))
    for path in (tmp_path/'pred').glob('*/*.jsonl'):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row['input_ids_sha256'] = 'changed' if path.parent.name == 'snap_256' else 'frozen'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = report(tmp_path)
    assert result.returncode != 0
    assert 'do not match frozen inputs' in result.stderr

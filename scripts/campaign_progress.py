"""Read-only coverage monitor; importing this script never initializes CUDA."""
import json
from pathlib import Path
import time


def progress():
    root = Path('results')
    status = json.loads((root/'campaign_status.json').read_text())
    child = status.get('child_pid')
    process = Path(f'/proc/{child}/cmdline') if child else None
    status['child_running'] = bool(process and process.exists() and
                                  status['phase'].encode() in process.read_bytes())
    status['checked_unix'] = time.time()
    run = root/'longbench_full'
    if (run/'protocol.json').exists():
        protocol = json.loads((run/'protocol.json').read_text())
        manifest = json.loads((run/'data_manifest.json').read_text())
        counts = {}
        by_task = {task: {} for task in protocol['tasks']}
        for method in protocol['methods']:
            name = method['name']
            counts[name] = 0
            for task in protocol['tasks']:
                path = run/'pred'/name/f'{task}.jsonl'
                count = path.read_bytes().count(b'\n') if path.exists() else 0
                counts[name] += count
                by_task[task][name] = count
        expected = sum(manifest['tasks'][task]['examples'] for task in protocol['tasks'])
        status.update(predictions=sum(counts.values()), expected_predictions=expected*len(counts),
                      minimum_completed_per_method=min(counts.values()), expected_per_method=expected,
                      counts=counts, completed_tasks=[task for task, values in by_task.items()
                      if all(count == manifest['tasks'][task]['examples'] for count in values.values())])
    failure = run/'failure.json'
    if failure.exists():
        status['recorded_failure'] = json.loads(failure.read_text())
    return status


if __name__ == '__main__':
    print(json.dumps(progress(), indent=2))

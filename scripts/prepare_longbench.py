"""Freeze every original LongBench test example using the pinned local sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from kv_distill.common import MODEL, REVISION

NO_CHAT = {'trec', 'triviaqa', 'samsum', 'lsht', 'lcc', 'repobench-p'}
LB_ROOT = Path('/home/qcb/kv-distill/third_party/LongBench/LongBench')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/longbench_full')
    parser.add_argument('--context-limit', type=int, default=32768)
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path('/home/qcb/kv-distill/configs/sources.json').read_text())
    revision = sources['datasets']['THUDM/LongBench']
    archive_path = hf_hub_download('THUDM/LongBench', 'data.zip', repo_type='dataset', revision=revision)
    archive = zipfile.ZipFile(archive_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    prompts = json.loads((LB_ROOT/'config/dataset2prompt.json').read_text())
    lengths = json.loads((LB_ROOT/'config/dataset2maxlen.json').read_text())
    pilot = json.loads(Path('results/quality_pilot/protocol.json').read_text())
    tuned_ids = {i for i in pilot['example_ids'] if i.startswith('longbench:')}
    manifest = dict(model=MODEL, model_revision=REVISION, dataset_revision=revision,
                    archive_sha256=sha(archive_path), context_limit=args.context_limit,
                    scope='All examples of all 21 original LongBench tasks; no LongBench-E rows',
                    prompt_protocol='Official task templates; Qwen chat template except official no-chat tasks; '
                    'token-level middle truncation after chat wrapping, preserving prefix and suffix; '
                    'reserve official max_new_tokens inside model context limit',
                    longbench_revision=subprocess.check_output(['git', '-C', str(LB_ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256={str(p): sha(p) for p in [Path(__file__), LB_ROOT/'config/dataset2prompt.json',
                                   LB_ROOT/'config/dataset2maxlen.json', LB_ROOT/'metrics.py', LB_ROOT/'eval.py']},
                    tasks={}, examples=[])
    for task, max_new in lengths.items():
        name = next(n for n in archive.namelist() if n == task+'.jsonl' or n.endswith('/'+task+'.jsonl'))
        rows = [json.loads(s) for s in archive.read(name).decode().splitlines()]
        path = dest/f'{task}.jsonl'
        with path.open('w') as f:
            for index, row in enumerate(rows):
                prompt = prompts[task].format(**row)
                ids = tokenizer.encode(prompt, add_special_tokens=False) if task in NO_CHAT else tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}], add_generation_prompt=True)
                original = len(ids)
                cap = args.context_limit-max_new
                if original > cap:
                    ids = ids[:cap//2] + ids[-(cap-cap//2):]
                example_id = f'longbench:{task}:{index}'
                record = dict(id=example_id, task=task, index=index, input_ids=ids, answers=row['answers'],
                              all_classes=row['all_classes'], length=row['length'], max_new_tokens=max_new,
                              original_tokens=original, prompt_tokens=len(ids), truncated=original>cap,
                              used_in_tuning=example_id in tuned_ids,
                              input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
                manifest['examples'].append({k: record[k] for k in ('id', 'task', 'prompt_tokens', 'original_tokens',
                                            'truncated', 'used_in_tuning', 'input_ids_sha256')})
        manifest['tasks'][task] = dict(examples=len(rows), sha256=sha(path), max_new_tokens=max_new)
        print(json.dumps(dict(task=task, examples=len(rows))), flush=True)
    manifest['total_examples'] = len(manifest['examples'])
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(total_examples=manifest['total_examples'], tasks=len(manifest['tasks']))), flush=True)


if __name__ == '__main__':
    main()

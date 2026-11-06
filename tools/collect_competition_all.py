"""Collect all 78 MUSE-A subsets/90 targets with selected raw external structures."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import zipfile

from collect_competition_samples import digest, json_bytes, require, signature
from competition_sample_assets import BundleWriter, collect_requests, export_assets, hash_file, source_path


SPLITS = ('train', 'validation', 'test')
DOMAINS = {'small_molecule', 'polymer', 'biomacromolecule', 'organic_crystal'}


def select_rows(path, declared, limit, seed, definitions):
    before = signature(path)
    require(before[0] == declared['bytes'], f'Source size differs from manifest: {path}')
    count, number = declared['records'], 0
    require(type(count) is int and count > 0, f'Invalid source count: {path}')
    wanted = set(random.Random(seed).sample(range(1, count + 1), min(limit, count)))
    hasher, selected, raw_lines, provenance, seen = hashlib.sha256(), [], [], [], set()
    labelled = {task: 0 for task in definitions}
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            require(number <= count, f'More rows than source manifest: {path}')
            hasher.update(raw)
            if number % 1000000 == 0:
                print(f'  scanned {number}/{count} rows: {path.parent.name}/{path.name}', flush=True)
            if number not in wanted:
                continue
            row = json.loads(raw)
            require(isinstance(row, dict) and isinstance(row.get('id'), str) and row['id'],
                    f'Invalid record ID: {path}:{number}')
            require(row['id'] not in seen, f'Duplicate sampled ID: {row["id"]}')
            seen.add(row['id'])
            labels = row.get('labels') or {}
            require(isinstance(labels, dict), f'Invalid label container: {row["id"]}')
            for task, definition in definitions.items():
                observations = labels.get(task) or []
                require(isinstance(observations, list), f'Invalid observations: {row["id"]}/{task}')
                for obs in observations:
                    require(isinstance(obs, dict) and 'value' in obs
                            and obs.get('task_type') in definition['task_types'],
                            f'Invalid observation schema: {row["id"]}/{task}')
                labelled[task] += bool(observations)
            selected.append(row)
            raw_lines.append(raw if raw.endswith(b'\n') else raw + b'\n')
            provenance.append({'id': row['id'], 'source_line': number, 'raw_line_sha256': digest(raw)})
    require(len(selected) == len(wanted) and number == count, f'Incomplete source: {path}')
    require(hasher.hexdigest() == declared['sha256'], f'Source hash differs from manifest: {path}')
    require(signature(path) == before, f'Source changed: {path}')
    return b''.join(raw_lines), selected, {
        'source_records': count, 'selected_records': len(selected), 'source_sha256': hasher.hexdigest(),
        'source_bytes': before[0], 'seed': seed, 'records_with_target_observations': labelled,
        'selected': provenance,
    }, before


def preflight(base, manifest, registry):
    subsets, task_ids, domains, paths = [], set(), set(), set()
    for dataset, entry in manifest['datasets'].items():
        for subset, definition in entry['subsets'].items():
            domain = definition['domain']
            require(domain in DOMAINS, f'Unknown domain: {domain}')
            relative = f'A_datasets/{domain}/{dataset}/{subset}'
            require(definition['directory'] == relative and definition['labels'], f'Unexpected subset: {relative}')
            for split in SPLITS:
                info = definition['files'][split]
                require(info['file'] == f'{relative}/{split}.jsonl', 'Manifest split path differs')
                path = source_path(base, 'data/' + info['file'])
                require(path not in paths, f'Duplicate source path: {path}')
                require(path.stat().st_size == info['bytes'], f'Source size differs: {path}')
                paths.add(path)
            domains.add(domain)
            task_ids.update(f'{dataset}/{subset}/{label}' for label in definition['labels'])
            subsets.append((domain, dataset, subset, definition))
    registered = [task['task_id'] for task in registry['tasks']]
    require(registry['board'] == 'A' and len(registered) == len(set(registered))
            and set(registered) == task_ids, 'Task registry and data manifest disagree')
    require(len(subsets) == 78 and len(task_ids) == 90 and domains == DOMAINS,
            'Expected 78 subsets and 90 targets across four domains; inspect changed release')
    require(importlib.util.find_spec('lmdb') is not None, 'lmdb is required to collect OMC25 raw rows')
    return subsets


def collect(args):
    base = args.base.resolve()
    output = args.output.resolve() if args.output else Path(__file__).resolve().parent / (
        'competition_all_samples_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.zip')
    partial = output.with_suffix(output.suffix + '.partial')
    require(output.suffix.lower() == '.zip' and not output.exists() and not partial.exists(), 'Use a NEW .zip path')
    require(not output.is_relative_to(base), 'Output must be outside source data')
    require(args.seed >= 0 and args.max_mb >= 0 and min(args.train, args.validation, args.test) > 0,
            'Invalid seed, limits or size cap')
    limits = dict(zip(SPLITS, (args.train, args.validation, args.test)))
    manifest_path = source_path(base, 'data/manifest.json')
    registry_path = args.task_registry.resolve()
    states = {manifest_path: signature(manifest_path), registry_path: signature(registry_path)}
    manifest_raw, registry_raw = manifest_path.read_bytes(), registry_path.read_bytes()
    manifest, registry = json.loads(manifest_raw), json.loads(registry_raw)
    subsets = preflight(base, manifest, registry)
    print('PREFLIGHT: 78 subsets / 90 targets / 4 domains. Original splits retained; no model/GPU.', flush=True)
    print('Uniform row sampling requires scanning source JSONL hashes, including large files.', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_reports, requests, coverage = {}, {}, []
    with tempfile.TemporaryDirectory(prefix='competition_collect_', dir=output.parent) as temp:
        work = Path(temp)
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            writer = BundleWriter(archive, args.max_mb * 1024**2)
            writer.add('metadata/original_manifest.json', manifest_raw)
            writer.add('metadata/tasks_A.json', registry_raw)
            for name in ('collect_competition_all.py', 'competition_sample_assets.py', 'collect_competition_samples.py'):
                writer.add('collector/' + name, Path(__file__).with_name(name).read_bytes())
            for domain, dataset, subset, definition in subsets:
                selected_ids = set()
                for split in SPLITS:
                    info = definition['files'][split]
                    relative = 'data/' + info['file']
                    path = source_path(base, relative)
                    seed = int(digest(f'{args.seed}:{relative}'.encode())[:16], 16)
                    print(f'SAMPLE {dataset}/{subset}/{split}: up to {limits[split]} of {info["records"]}', flush=True)
                    raw, rows, report, stat = select_rows(path, info, limits[split], seed, definition['labels'])
                    ids = {row['id'] for row in rows}
                    require(not selected_ids.intersection(ids), f'Sampled IDs overlap original splits: {dataset}/{subset}')
                    selected_ids.update(ids)
                    # Missing observations are counted, never silently dropped or replaced.
                    if split == 'train':
                        require(all(v > 0 for v in report['records_with_target_observations'].values()),
                                f'Sampled train rows do not cover every target: {dataset}/{subset}')
                    writer.add(relative, raw)
                    collect_requests(rows, domain, dataset, requests)
                    states[path], source_reports[relative] = stat, report
                    coverage.append({'domain': domain, 'dataset': dataset, 'subset': subset, 'split': split,
                                     'selected_records': len(rows), 'targets': report['records_with_target_observations']})
            print(f'JSONL sampling complete. Collecting {len(requests)} referenced structure containers/files.', flush=True)
            structures = export_assets(base, requests, writer, work)
            require(all(signature(path) == state for path, state in states.items()), 'Source files changed during export')
            for relative, item in structures.items():
                require(signature(source_path(base, relative)) == (item['source_bytes'], item['source_mtime_ns']),
                        f'Structure source changed after collection: {relative}')
            report = {
                'status': 'all_subsets_sampled_with_structures', 'subsets': 78, 'targets': 90,
                'domains': sorted(DOMAINS), 'limits': limits, 'seed': args.seed, 'coverage': coverage,
                'sources': source_reports, 'structures': structures, 'files': dict(writer.files),
                'original_manifest_sha256': digest(manifest_raw), 'task_registry_sha256': digest(registry_raw),
                'sampling': 'Uniform sample without replacement per original split; retain original row order and all observations.',
                'scope': 'Collection verification only. No invalid chemistry filtered or repaired. '
                         'Class coverage, finite labels, chemical alignment and pretraining overlap NOT audited. '
                         'Some tasks need pair/token/3D heads; this is not proof the tiny sequence model supports all 90 tasks. '
                         'LMDB row keys/values are preserved in sparse databases; full ASE database metadata is not retained. '
                         'Original full-data manifest/registry counts and baseline statistics must NOT be used as sample counts or fitted statistics.'}
            writer.add('sample_manifest.json', json_bytes(report))
            members = dict(writer.files)
        print('VERIFY: reading ZIP members and checking every collected SHA256...', flush=True)
        with zipfile.ZipFile(partial) as archive:
            require(len(archive.namelist()) == len(members) and set(archive.namelist()) == set(members), 'ZIP members differ')
            for name, expected in members.items():
                hasher, size = hashlib.sha256(), 0
                with archive.open(name) as stream:
                    while chunk := stream.read(8 * 1024**2):
                        hasher.update(chunk)
                        size += len(chunk)
                require(size == expected['bytes'] and hasher.hexdigest() == expected['sha256'], f'ZIP mismatch: {name}')
    require(not output.exists(), 'Final output already exists')
    partial.rename(output)
    print(f'READY: {output} ({output.stat().st_size / 1024**2:.2f} MiB)', flush=True)
    print(f'SHA256: {hash_file(output)}')
    print('Coverage: 78 subsets / 90 targets / 4 domains. Selected external structures included. No training performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/yinghepool/tech-dcu/zhaoyao/data/muse-a/MUSE-A'))
    parser.add_argument('--task-registry', type=Path, default=Path('/yinghepool/tech-dcu/zhaoyao/muse/tasks_A.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--validation', type=int, default=300)
    parser.add_argument('--test', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-mb', type=int, default=1024, help='Uncompressed bundle limit; 0 disables limit')
    collect(parser.parse_args())

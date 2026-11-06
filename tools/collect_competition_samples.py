"""Server-side stdlib-only sampling; no GPU, training, extraction or source edits."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import zipfile


TASKS = (
    ('small_molecule', 'moleculenet', 'freesolv', 'freesolv__y'),
    ('polymer', 'TransPolymer', 'EPS', 'eps'),
    ('biomacromolecule', 'PEER', 'gb1', 'gb1_fitness'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def inventory(args):
    base = args.base.resolve()
    output = args.output.resolve() if args.output else Path(__file__).resolve().parent / (
        'competition_inventory_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    require(output.suffix.lower() == '.json' and not output.exists(), 'Use a new .json output path')
    require(not output.is_relative_to(base), 'Inventory output must be outside source data')
    manifest_path = base / 'data/manifest.json'
    manifest_stat = signature(manifest_path)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    rows, external_files, missing, domains = [], {}, [], {}
    registered = set()
    for dataset, data in manifest['datasets'].items():
        for subset, definition in data['subsets'].items():
            domain = definition['domain']
            directory = (base / 'data' / definition['directory']).resolve()
            require(directory.is_relative_to(base / 'data'), f'Unsafe subset directory: {directory}')
            require(directory not in registered, f'Duplicate subset directory: {directory}')
            registered.add(directory)
            domains[domain] = domains.get(domain, 0) + 1
            entry = {'dataset': dataset, 'subset': subset, 'domain': domain, 'definition': definition, 'splits': {}}
            for split in ('train', 'validation', 'test'):
                path = directory / f'{split}.jsonl'
                if not path.is_file():
                    missing.append(str(path))
                    entry['splits'][split] = {'missing': True}
                    continue
                size, mtime = signature(path)
                examples = []
                with path.open('rb') as stream:
                    for number in range(1, 4):
                        line = stream.readline()
                        if not line:
                            break
                        record = json.loads(line)
                        examples.append({'source_line': number, 'id': record['id'],
                                         'sample_type': record.get('sample_type'),
                                         'input_keys': sorted((record.get('inputs') or {}).keys()),
                                         'label_names': sorted((record.get('labels') or {}).keys()),
                                         'external_inputs': record.get('external_inputs') or []})
                require(signature(path) == (size, mtime), f'File changed during inspection: {path}')
                entry['splits'][split] = {'bytes': size, 'prefix_examples': examples}
            external_base = (base / 'external_inputs/A_datasets' / domain / dataset).resolve()
            require(external_base.is_relative_to(base / 'external_inputs'), 'Unsafe external directory')
            if str(external_base) not in external_files:
                external_files[str(external_base)] = [
                    {'path': p.relative_to(base).as_posix(), 'bytes': p.stat().st_size}
                    for p in sorted(external_base.rglob('*')) if p.is_file()]
            rows.append(entry)
            mib = sum(s.get('bytes', 0) for s in entry['splits'].values()) / 1024**2
            print(f'{domain}/{dataset}/{subset}: {len(definition.get("labels", {}))} targets; '
                  f'JSONL={mib:.2f} MiB; external_formats={definition.get("external_formats", {})}', flush=True)
    unregistered = sorted(str(p.parent) for p in (base / 'data/A_datasets').rglob('train.jsonl')
                          if p.parent.resolve() not in registered)
    require(signature(manifest_path) == manifest_stat, 'Manifest changed during inspection')
    report = {'status': 'inventory_only', 'source_base': str(base), 'domains': domains,
              'subsets': rows, 'external_files': external_files, 'missing_split_files': missing,
              'directories_not_registered_in_manifest': unregistered,
              'lmdb_available': importlib.util.find_spec('lmdb') is not None,
              'source_manifest': manifest, 'source_manifest_sha256': digest(raw),
              'scope': 'Metadata and first 3 records per split only. No full data scan, extraction, '
                       'structure validation, model loading or sample bundle. Prefix examples do not enumerate all references.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(json_bytes(report))
    print(f'INVENTORY ONLY: {output}')
    print(f'Domains={domains}; subsets={len(rows)}; missing_files={len(missing)}; unregistered={len(unregistered)}')
    print('No all-dataset ZIP produced yet; send this inventory before collecting external structures.')


def sample_file(path, limit, seed, split, label, seen_ids):
    before = signature(path)
    hasher, rng, reservoir = hashlib.sha256(), random.Random(seed), []
    total = 0
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            hasher.update(raw)
            require(raw.strip(), f'Blank JSONL line: {path}:{number}')
            row = json.loads(raw)
            require(isinstance(row, dict), f'Not an object: {path}:{number}')
            key = row.get('id')
            require(isinstance(key, str) and key, f'Missing string ID: {path}:{number}')
            require(key not in seen_ids, f'Duplicate ID within/across original splits: {key}')
            seen_ids.add(key)
            total += 1
            item = (number, raw, row)
            if len(reservoir) < limit:
                reservoir.append(item)
            else:
                slot = rng.randrange(total)
                if slot < limit:
                    reservoir[slot] = item
    require(signature(path) == before, f'Source changed while reading: {path}')
    require(total > 0, f'Empty source: {path}')
    reservoir.sort(key=lambda item: item[0])
    selected, labelled, keys = [], 0, set()
    for number, raw, row in reservoir:
        require(isinstance(row.get('inputs'), dict) and row['inputs'], f'Missing inputs: {row["id"]}')
        require(not row.get('external_inputs'),
                f'External files required by {row["id"]}; stop and inspect before exporting')
        keys.update(row['inputs'])
        labels = row.get('labels') or {}
        require(isinstance(labels, dict), f'Invalid labels: {row["id"]}')
        observations = labels.get(label)
        if observations:
            require(isinstance(observations, list) and all(
                isinstance(obs, dict) and obs.get('task_type') == 'regression' and 'value' in obs
                for obs in observations), f'Unexpected label schema: {row["id"]}/{label}')
            labelled += 1
        else:
            require(split == 'test', f'Missing {split} label: {row["id"]}/{label}')
        selected.append({'id': row['id'], 'source_line': number, 'record_sha256': digest(raw)})
    payload = b''.join(raw if raw.endswith(b'\n') else raw + b'\n' for _, raw, _ in reservoir)
    return payload, {
        'source_path': str(path), 'source_bytes': before[0], 'source_sha256': hasher.hexdigest(),
        'source_records': total, 'requested_max': limit, 'selected_records': len(reservoir),
        'records_with_target_observations': labelled, 'input_keys': sorted(keys),
        'target': label, 'seed': seed, 'selected': selected,
        'note': 'Original JSON record bytes retained; only a missing terminal newline is appended.'
    }, before


def collect(args):
    base = args.base.resolve()
    output = args.output.resolve() if args.output else Path(__file__).resolve().parent / (
        'competition_samples_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.zip')
    partial = output.with_suffix(output.suffix + '.partial')
    require(output.suffix.lower() == '.zip', 'Output must end in .zip')
    require(not output.exists() and not partial.exists(), f'Output already exists: {output}')
    require(not output.is_relative_to(base), 'Output must be outside the source dataset directory')
    limits = {'train': args.train, 'validation': args.validation, 'test': args.test}
    require(all(n > 0 for n in limits.values()) and args.seed >= 0, 'Limits must be positive; seed nonnegative')
    manifest_path = base / 'data/manifest.json'
    manifest_stat = signature(manifest_path)
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    files = {'metadata/source_manifest.json': manifest_raw}
    sources, task_definitions, source_stats = {}, {}, {manifest_path: manifest_stat}
    for domain, dataset, subset, label in TASKS:
        relative = f'A_datasets/{domain}/{dataset}/{subset}'
        definition = manifest['datasets'][dataset]['subsets'][subset]
        require(definition['directory'].replace('\\', '/') == relative
                and definition['domain'] == domain, f'Manifest directory mismatch: {dataset}/{subset}')
        require(set(definition['labels'][label]['task_types']) == {'regression'},
                f'Not the expected regression task: {dataset}/{subset}/{label}')
        task_definitions[f'{dataset}/{subset}'] = definition
        seen_ids = set()
        for split, limit in limits.items():
            member = f'data/{relative}/{split}.jsonl'
            path = (base / member).resolve()
            require(path.is_relative_to(base), f'Source escapes base directory: {path}')
            split_seed = int(digest(f'{args.seed}:{member}'.encode('utf-8'))[:16], 16)
            payload, details, stat = sample_file(path, limit, split_seed, split, label, seen_ids)
            files[member], sources[member], source_stats[path] = payload, details, stat
            print(f'{dataset}/{subset}/{split}: selected={details["selected_records"]}/{details["source_records"]}; '
                  f'with_target={details["records_with_target_observations"]}', flush=True)
    require(all(signature(path) == stat for path, stat in source_stats.items()), 'Source changed during export')
    report = {'status': 'sample_export_only', 'source_base': str(base), 'seed': args.seed,
              'limits': limits, 'sources': sources, 'original_task_definitions': task_definitions,
              'source_manifest_sha256': digest(manifest_raw),
              'script_sha256': digest(Path(__file__).read_bytes()),
              'member_sha256': {name: digest(data) for name, data in files.items()},
              'scope': 'Original split assignments retained. All label observations retained without aggregation. '
                       'Source manifest describes FULL source data, not sample counts. '
                       'No chemical/tokenization/finite-label/pretraining-overlap audit or training. '
                       'Test labels may be absent; this is not a full official benchmark.'}
    files['sample_manifest.json'] = json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with zipfile.ZipFile(partial) as archive:
        require(archive.testzip() is None and set(archive.namelist()) == set(files), 'ZIP integrity check failed')
        for name, data in files.items():
            require(digest(archive.read(name)) == digest(data), f'ZIP member differs: {name}')
    require(all(signature(path) == stat for path, stat in source_stats.items()), 'Source changed before completion')
    require(not output.exists(), f'Output appeared during export: {output}')
    partial.rename(output)
    print(f'READY: {output} ({output.stat().st_size / 1024**2:.2f} MiB)')
    print(f'SHA256: {digest(output.read_bytes())}')
    print('Sampling and ZIP checks only; training-readiness remains to be checked locally.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/yinghepool/tech-dcu/zhaoyao/data/muse-a/MUSE-A'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--validation', type=int, default=300)
    parser.add_argument('--test', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--inventory', action='store_true', help='Inspect ALL manifest subsets without collecting data')
    args = parser.parse_args()
    if args.inventory:
        inventory(args)
    else:
        collect(args)

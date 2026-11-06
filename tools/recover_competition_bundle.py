"""Recover a bundle interrupted at final manifest writing; preserve the input ZIP."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

from collect_competition_all import DOMAINS, SPLITS, preflight, select_rows
from collect_competition_samples import digest, json_bytes, require, signature
from competition_sample_assets import BundleWriter, collect_requests, hash_file, safe_relative, source_path


COLLECTORS = ('collect_competition_all.py', 'competition_sample_assets.py',
              'collect_competition_samples.py')


def stream_identity(stream):
    hasher, size = hashlib.sha256(), 0
    while chunk := stream.read(8 * 1024**2):
        hasher.update(chunk)
        size += len(chunk)
    return {'bytes': size, 'sha256': hasher.hexdigest()}


def check_lmdb(stream, expected_size, wanted, work):
    import lmdb

    with tempfile.TemporaryDirectory(prefix='recovery_lmdb_', dir=work) as temp:
        path = Path(temp) / 'sample.aselmdb'
        with path.open('xb') as dest:
            shutil.copyfileobj(stream, dest, 8 * 1024**2)
        require(path.stat().st_size == expected_size, 'Truncated packaged LMDB')
        env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
        try:
            with env.begin() as txn:
                expected_keys = {str(index).encode('ascii') for index in wanted}
                require(set(txn.cursor().iternext(values=False)) == expected_keys,
                        'Packaged LMDB keys differ from requested rows')
                rows = []
                for index in sorted(wanted):
                    value = txn.get(str(index).encode('ascii'))
                    require(value is not None and len(value) > 0, f'Empty LMDB row: {index}')
                    rows.append({'lmdb_row_id': index, 'raw_value_sha256': digest(value)})
        finally:
            env.close()
    return {'rows': rows, 'representation': 'Sparse original numeric keys; no full ASE metadata.'}


def check_structures(archive, requests, work):
    result = {}
    for relative, request in sorted(requests.items()):
        fmt, wanted = request['format'], request['selectors']
        entries, found = [], set()
        print(f'CHECK PACKAGED STRUCTURE {fmt}: {relative}', flush=True)
        with archive.open(relative) as stream:
            if fmt == 'sdf':
                block = []
                for line in stream:
                    block.append(line)
                    if line.strip() != b'$$$$':
                        continue
                    key = block[0].decode('utf-8').strip()
                    require(key in wanted and key not in found, f'Unexpected/duplicate SDF ID: {key}')
                    entries.append({'record_id': key, 'sha256': digest(b''.join(block))})
                    found.add(key)
                    block = []
                require(not block and found == set(wanted), f'Incomplete SDF references: {relative}')
            elif fmt in ('rdkit_pickle', 'ase_lmdb'):
                with tarfile.open(fileobj=stream, mode='r|') as source:
                    for member in source:
                        safe_relative(member.name, member=True)
                        require(member.isfile() and not member.linkname and member.size > 0
                                and member.name in wanted and member.name not in found,
                                f'Unexpected/invalid tar member: {member.name}')
                        with source.extractfile(member) as content:
                            if fmt == 'ase_lmdb':
                                detail = check_lmdb(content, member.size, wanted[member.name], work)
                            else:
                                identity = stream_identity(content)
                                require(identity['bytes'] == member.size, 'Truncated packaged pickle')
                                detail = {'raw_sha256': identity['sha256'],
                                          'requested_conformer_indices': sorted(wanted[member.name]),
                                          'scope': 'Not unpickled; conformer index semantics not checked.'}
                        entries.append({'archive_member': member.name, **detail})
                        found.add(member.name)
                require(found == set(wanted), f'Missing tar references: {relative}')
                # Consume any remaining padding so the outer ZIP CRC is also checked.
                while stream.read(8 * 1024**2):
                    pass
            else:
                require(fmt == 'cif', f'Unsupported format: {fmt}')
                identity = stream_identity(stream)
                require(identity['bytes'] > 0, f'Empty CIF: {relative}')
                entries.append({'raw_sha256': identity['sha256']})
        result[relative] = {'format': fmt, 'entries': entries,
                            'scope': 'Packaged reference coverage checked; original structure source not re-compared.'}
    return result


def recover(args):
    partial, base = args.partial.resolve(), args.base.resolve()
    require(partial.is_file() and partial.name.endswith('.zip.partial'), 'Expected existing .zip.partial')
    require(args.seed >= 0 and min(args.train, args.validation, args.test) > 0, 'Invalid sampling settings')
    output = partial.with_name(partial.name[:-len('.zip.partial')] + '_recovered.zip')
    writing = output.with_suffix('.zip.partial')
    sidecar = output.with_suffix('.manifest.json')
    require(not output.is_relative_to(base), 'Recovery output must be outside original data')
    require(all(not path.exists() for path in (output, writing, sidecar)), 'Recovery output already exists; preserve it')
    before = signature(partial)
    limits = dict(zip(SPLITS, (args.train, args.validation, args.test)))
    with zipfile.ZipFile(partial) as original:
        names = original.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP members')
        require('sample_manifest.json' in names, 'Not a final-manifest interruption; do not recover blindly')
        try:
            json.loads(original.read('sample_manifest.json'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        else:
            raise ValueError('Manifest is already valid JSON; inspect instead of overwriting it')
        for name in names:
            safe_relative(name)
        for name in COLLECTORS:
            require(original.read('collector/' + name) == Path(__file__).with_name(name).read_bytes(),
                    f'Collector differs from bundled version: {name}; inspect before recovery')
        manifest_raw = original.read('metadata/original_manifest.json')
        registry_raw = original.read('metadata/tasks_A.json')
        manifest_path = source_path(base, 'data/manifest.json')
        require(manifest_path.read_bytes() == manifest_raw, 'Original data manifest has changed')
        manifest, registry = json.loads(manifest_raw), json.loads(registry_raw)
        subsets = preflight(base, manifest, registry)
        requests, sources, coverage, states = {}, {}, [], {manifest_path: signature(manifest_path)}
        expected = {'metadata/original_manifest.json', 'metadata/tasks_A.json', 'sample_manifest.json'}
        expected.update('collector/' + name for name in COLLECTORS)
        for domain, dataset, subset, definition in subsets:
            selected_ids = set()
            for split in SPLITS:
                declared = definition['files'][split]
                relative = 'data/' + declared['file']
                expected.add(relative)
                require(relative in names, f'Missing sampled split: {relative}')
                seed = int(digest(f'{args.seed}:{relative}'.encode())[:16], 16)
                print(f'RECHECK SOURCE {dataset}/{subset}/{split}', flush=True)
                path = source_path(base, relative)
                raw, rows, report, stat = select_rows(path, declared, limits[split], seed, definition['labels'])
                require(original.read(relative) == raw, f'Sampled bytes differ from original source: {relative}')
                ids = {row['id'] for row in rows}
                require(not selected_ids.intersection(ids), f'Sampled split IDs overlap: {relative}')
                selected_ids.update(ids)
                if split == 'train':
                    require(all(v > 0 for v in report['records_with_target_observations'].values()),
                            f'Missing training target coverage: {relative}')
                sources[relative], states[path] = report, stat
                collect_requests(rows, domain, dataset, requests)
                coverage.append({'domain': domain, 'dataset': dataset, 'subset': subset, 'split': split,
                                 'selected_records': len(rows), 'targets': report['records_with_target_observations']})
        expected.update(requests)
        require(set(names) == expected, 'Missing or unexpected ZIP payload members')
        with tempfile.TemporaryDirectory(prefix='competition_recovery_', dir=output.parent) as temp:
            structures = check_structures(original, requests, Path(temp))
        require(all(signature(path) == state for path, state in states.items()), 'Original JSONL/manifest changed')
        print('COPY: writing a NEW ZIP; original partial is retained. No structure source collection.', flush=True)
        with zipfile.ZipFile(writing, 'x', compression=zipfile.ZIP_DEFLATED) as dest:
            writer = BundleWriter(dest, 0)
            for name in names:
                if name == 'sample_manifest.json':
                    continue
                with original.open(name) as stream:
                    writer.add_stream(name, stream)
            writer.add('recovery/recover_competition_bundle.py', Path(__file__).read_bytes())
            input_hash = hash_file(partial)
            require(signature(partial) == before, 'Input partial changed during recovery')
            report = {
                'status': 'recovered_collection_internal_checks_passed',
                'subsets': 78, 'targets': 90, 'domains': sorted(DOMAINS),
                'limits': limits, 'seed': args.seed, 'coverage': coverage,
                'sources': sources, 'structures': structures, 'files': dict(writer.files),
                'original_manifest_sha256': digest(manifest_raw), 'task_registry_sha256': digest(registry_raw),
                'recovery': {'input': str(partial), 'input_sha256': input_hash,
                             'timestamp': datetime.now().isoformat(),
                             'source_jsonl_rehashed_and_sampled_bytes_compared': True,
                             'original_structure_bytes_recompared': False,
                             'original_structure_source_hashes_recovered': False,
                             'bundle_size_limit_disabled_for_recovery': True},
                'scope': '78 subsets / 90 target definitions / 234 splits. Original JSONL sources and sampled bytes verified. '
                         'Packaged SDF IDs, tar members and sparse LMDB keys checked; copied payload CRC/SHA checked. '
                         'Lost original structure-source hashes cannot be reconstructed from the ZIP and are NOT claimed. '
                         'Original structure files were NOT re-compared. Pickles NOT unpickled; conformer indices NOT validated. '
                         'No chemical alignment, finite-label, class coverage, leakage, tokenizer or training-readiness audit. '
                         'Full-data metadata counts are NOT sample counts. Sparse LMDB has no full ASE metadata.'}
            report_bytes = json_bytes(report)
            # Persist the report first: another finalization failure must not lose it again.
            with sidecar.open('xb') as stream:
                stream.write(report_bytes)
            writer.add('sample_manifest.json', report_bytes)
            members = dict(writer.files)
    print('VERIFY: every new ZIP member SHA256 and byte count...', flush=True)
    with zipfile.ZipFile(writing) as archive:
        require(len(archive.namelist()) == len(members) and set(archive.namelist()) == set(members), 'Output members differ')
        for name, expected_identity in members.items():
            with archive.open(name) as stream:
                require(stream_identity(stream) == expected_identity, f'Output content mismatch: {name}')
    require(signature(partial) == before and not output.exists(), 'Input changed or final output exists')
    writing.rename(output)
    print(f'READY: {output} ({output.stat().st_size / 1024**2:.2f} MiB)', flush=True)
    print(f'SHA256: {hash_file(output)}', flush=True)
    print('234 splits / 78 subsets / 90 targets; packaged references checked; original partial retained.')
    print('Original structure bytes NOT re-compared; chemistry/training acceptance still pending.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', required=True, type=Path)
    parser.add_argument('--base', type=Path, default=Path('/yinghepool/tech-dcu/zhaoyao/data/muse-a/MUSE-A'))
    parser.add_argument('--train', type=int, default=2000)
    parser.add_argument('--validation', type=int, default=300)
    parser.add_argument('--test', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    recover(parser.parse_args())

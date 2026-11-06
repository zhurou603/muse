"""Copy selected raw structures; never unpickle or chemically rewrite records."""
import hashlib
import io
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from collect_competition_samples import digest, require, signature


def safe_relative(value, *, member=False):
    require(isinstance(value, str) and value, 'Empty relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and '\x00' not in value,
            f'Unsafe path: {value}')
    require(member or ('\\' not in value and ':' not in value), f'Unsafe filesystem path: {value}')
    return path.as_posix()


def source_path(base, relative):
    path = (base / safe_relative(relative)).resolve()
    require(path.is_relative_to(base.resolve()) and path.is_file(), f'Missing/unsafe source: {path}')
    return path


def hash_file(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


class BundleWriter:
    def __init__(self, archive, max_bytes):
        self.archive, self.max_bytes = archive, max_bytes
        self.total, self.files = 0, {}

    def add_stream(self, name, stream):
        name = safe_relative(name)
        require(name not in self.files, f'Duplicate ZIP member: {name}')
        hasher, size = hashlib.sha256(), 0
        with self.archive.open(name, 'w', force_zip64=True) as dest:
            while chunk := stream.read(8 * 1024**2):
                self.total += len(chunk)
                require(not self.max_bytes or self.total <= self.max_bytes,
                        'Bundle exceeds configured uncompressed size limit; no complete ZIP produced')
                hasher.update(chunk)
                size += len(chunk)
                dest.write(chunk)
        self.files[name] = {'bytes': size, 'sha256': hasher.hexdigest()}

    def add(self, name, data):
        self.add_stream(name, io.BytesIO(data))


def collect_requests(rows, domain, dataset, requests):
    prefix = f'external_inputs/A_datasets/{domain}/{dataset}'
    for row in rows:
        for ref in row.get('external_inputs') or []:
            fmt = ref['format']
            require(fmt in ('sdf', 'rdkit_pickle', 'cif', 'ase_lmdb'), f'Unsupported structure format: {fmt}')
            relative = prefix + '/' + safe_relative(ref['source_path'])
            request = requests.setdefault(relative, {'format': fmt, 'selectors': {}})
            require(request['format'] == fmt, f'Conflicting formats: {relative}')
            selector = ref.get('selector') or {}
            if fmt == 'sdf':
                key = selector['record_id']
                require(isinstance(key, str) and key, 'Invalid SDF record_id')
                request['selectors'][key] = True
            elif fmt in ('rdkit_pickle', 'ase_lmdb'):
                member = safe_relative(selector['archive_member'], member=True)
                wanted = request['selectors'].setdefault(member, set())
                index = selector['conformer_index'] if fmt == 'rdkit_pickle' else selector['lmdb_row_id']
                require(type(index) is int and index >= (0 if fmt == 'rdkit_pickle' else 1),
                        f'Invalid structure row/index: {row["id"]}')
                wanted.add(index)
            else:
                require(not selector, f'Unsupported CIF selector: {selector}')


def copy_sdf(path, wanted, writer, relative, work):
    found, rows = set(), []
    with tempfile.TemporaryFile(dir=work) as sampled, path.open('rb') as source:
        block = []
        for line in source:
            block.append(line)
            if line.strip() != b'$$$$':
                continue
            key = block[0].decode('utf-8').strip()
            if key in wanted:
                require(key not in found, f'Duplicate selected SDF ID: {key}')
                raw = b''.join(block)
                sampled.write(raw)
                rows.append({'record_id': key, 'sha256': digest(raw)})
                found.add(key)
            block = []
            if found == wanted:
                break
        require(found == wanted, f'Missing {len(wanted - found)} SDF structures in {path}')
        sampled.seek(0)
        writer.add_stream(relative, sampled)
    return rows


def copy_lmdb_member(source, size, wanted, work):
    import lmdb

    require(shutil.disk_usage(work).free > size + 128 * 1024**2,
            f'Not enough temporary disk for one LMDB member ({size} bytes)')
    original, reduced = work / 'original.aselmdb', work / 'sample.aselmdb'
    hasher, copied = hashlib.sha256(), 0
    with original.open('xb') as dest:
        next_progress = 256 * 1024**2
        while chunk := source.read(8 * 1024**2):
            hasher.update(chunk)
            copied += len(chunk)
            dest.write(chunk)
            if copied >= next_progress:
                print(f'    temporary LMDB read: {copied / 1024**2:.0f}/{size / 1024**2:.0f} MiB', flush=True)
                next_progress += 256 * 1024**2
    require(copied == size, 'Truncated LMDB archive member')
    values = {}
    env = lmdb.open(str(original), subdir=False, readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            for row_id in sorted(wanted):
                key = str(row_id).encode('ascii')
                value = txn.get(key)
                require(value is not None, f'Missing LMDB row: {row_id}')
                values[key] = value
    finally:
        env.close()
    map_size = max(16 * 1024**2, sum(len(v) for v in values.values()) * 4 + 16 * 1024**2)
    env = lmdb.open(str(reduced), subdir=False, map_size=map_size)
    try:
        with env.begin(write=True) as txn:
            for key, value in values.items():
                require(txn.put(key, value, overwrite=False), f'Duplicate LMDB key: {key}')
        env.sync()
    finally:
        env.close()
    env = lmdb.open(str(reduced), subdir=False, readonly=True, lock=False)
    try:
        with env.begin() as txn:
            require(txn.stat()['entries'] == len(values), 'Unexpected sparse LMDB entries')
            require(all(txn.get(k) == v for k, v in values.items()), 'Sparse LMDB raw values changed')
    finally:
        env.close()
    return reduced, {'source_member_sha256': hasher.hexdigest(), 'rows': [
        {'lmdb_row_id': int(k), 'raw_value_sha256': digest(v)} for k, v in values.items()],
        'representation': 'Sparse LMDB: original numeric keys and raw values; no full-database ASE metadata.'}


def copy_tar(path, request, writer, relative, work):
    wanted, found, details = request['selectors'], set(), []
    with tempfile.TemporaryFile(dir=work) as sampled:
        with tarfile.open(path, 'r:') as source, tarfile.open(fileobj=sampled, mode='w') as dest:
            for member in source:
                if member.name not in wanted:
                    continue
                safe_relative(member.name, member=True)
                require(member.isfile() and not member.linkname and member.name not in found,
                        f'Invalid/duplicate tar member: {member.name}')
                print(f'  member {member.name}: {member.size / 1024**2:.1f} MiB source', flush=True)
                with source.extractfile(member) as stream:
                    if request['format'] == 'ase_lmdb':
                        with tempfile.TemporaryDirectory(prefix='lmdb_', dir=work) as temp:
                            reduced, info = copy_lmdb_member(stream, member.size, wanted[member.name], Path(temp))
                            header = tarfile.TarInfo(member.name)
                            header.size = reduced.stat().st_size
                            with reduced.open('rb') as rows:
                                dest.addfile(header, rows)
                    else:
                        require(not writer.max_bytes or writer.total + sampled.tell() + member.size <= writer.max_bytes,
                                'Selected pickle members exceed bundle size limit')
                        raw = stream.read()
                        require(len(raw) == member.size, 'Truncated pickle member')
                        dest.addfile(member, io.BytesIO(raw))
                        info = {'raw_sha256': digest(raw), 'requested_conformer_indices': sorted(wanted[member.name]),
                                'scope': 'Entire original pickle retained; not unpickled or chemically validated.'}
                details.append({'archive_member': member.name, **info})
                found.add(member.name)
                if found == set(wanted):
                    break
        require(found == set(wanted), f'Missing tar members: {path}; count={len(set(wanted) - found)}')
        sampled.seek(0)
        writer.add_stream(relative, sampled)
    return details


def export_assets(base, requests, writer, work):
    ledger = {}
    for relative, request in sorted(requests.items()):
        path = source_path(base, relative)
        before = signature(path)
        fmt = request['format']
        print(f'STRUCTURE {fmt}: {relative}', flush=True)
        if fmt == 'sdf':
            entries = copy_sdf(path, set(request['selectors']), writer, relative, work)
        elif fmt in ('rdkit_pickle', 'ase_lmdb'):
            entries = copy_tar(path, request, writer, relative, work)
        else:
            with path.open('rb') as stream:
                writer.add_stream(relative, stream)
            entries = [{'raw_sha256': writer.files[relative]['sha256']}]
        require(signature(path) == before, f'Structure source changed: {path}')
        ledger[relative] = {'format': fmt, 'source_bytes': before[0], 'source_mtime_ns': before[1],
                            'entries': entries, 'scope': 'Selected structure bytes verified; full source file not hashed.'}
    return ledger

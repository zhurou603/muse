"""Install the verified fixed-card candidate, or restore its original backup."""
import argparse
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import stat
import sys
import tempfile

ORIGINAL = '2a9afbd3e51900bb4306250a8e7a4ab9343d7096af8ce6f5b018b6cbc8a07a0a'
FIXED = '2f3e3dcdd8c550a014f7ca696697809eb26f1f0b2a16c7270bdede66e6bfe1d6'
DEFAULT_ROOT = Path('/yinghepool/tech-dcu/zhaoyao')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = path.read_bytes()
    actual = digest(data)
    if actual != expected:
        raise RuntimeError(f'Hash mismatch: {path}; expected={expected}, actual={actual}; stopped')
    return data


def replace_checked(target, before_hash, content, after_hash):
    if digest(content) != after_hash:
        raise RuntimeError('Replacement content differs; stopped')
    checked(target, before_hash)
    permissions = stat.S_IMODE(target.stat().st_mode)
    fd, name = tempfile.mkstemp(prefix='.parallel_dashboard.fixed_card.', suffix='.tmp', dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, permissions)
        checked(temporary, after_hash)
        checked(target, before_hash)
        os.replace(temporary, target)
        checked(target, after_hash)
    finally:
        if temporary.exists():
            temporary.unlink()


def install(root, execute=False, rollback=False):
    target = root / 'model_design/muse_encoder/pretrain/parallel_dashboard.py'
    package = root / 'tools/fixed_card_patch_20261102'
    backup = package / 'installed_backup/parallel_dashboard.py'
    if target.is_symlink():
        raise RuntimeError('Production file is a symlink; stopped')
    if rollback:
        before_hash, after_hash = FIXED, ORIGINAL
        replacement = checked(backup, ORIGINAL)
    else:
        before_hash, after_hash = ORIGINAL, FIXED
        replacement = checked(package / 'candidate/fixed_parallel_dashboard.py', FIXED)
    current = checked(target, before_hash)
    compile(replacement, str(target), 'exec')
    if backup.exists():
        checked(backup, ORIGINAL)
    print(f'CHECK PASS: {"rollback" if rollback else "install"}; target={target}', flush=True)
    if not execute:
        print('CHECK ONLY: no production change. Use --execute to apply.', flush=True)
        return target
    if not rollback and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open('xb') as stream:
            stream.write(current)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(backup, stat.S_IMODE(target.stat().st_mode))
        checked(backup, ORIGINAL)
    print(f'BACKUP: {backup}', flush=True)
    replace_checked(target, before_hash, replacement, after_hash)
    print(f'RESTORED: {target}' if rollback else f'INSTALLED: {target}', flush=True)
    print(f'SHA256: {after_hash}', flush=True)
    return target


def verify_import(root, target, rollback):
    sys.dont_write_bytecode = True
    os.chdir(root)
    sys.path.insert(0, str(root / 'model_design'))
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    module = importlib.import_module('muse_encoder.pretrain.parallel_dashboard')
    if Path(module.__file__).resolve() != target.resolve():
        raise RuntimeError(f'Imported unexpected file: {module.__file__}')
    checked(target, ORIGINAL if rollback else FIXED)
    fixed_interface = 'devices' in inspect.signature(module._parallel_map).parameters
    if fixed_interface != (not rollback):
        raise RuntimeError('Imported interface differs from expected version')
    if not rollback and not callable(getattr(module, '_init_encoding_worker', None)):
        raise RuntimeError('Fixed worker initializer missing')
    print(f'IMPORT PASS: {module.__file__}; no training started', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    target = install(root, args.execute, args.rollback)
    if args.execute:
        try:
            verify_import(root, target, args.rollback)
        except BaseException:
            print('IMPORT CHECK FAILED after replacement. Backup retained; do not start evaluation.',
                  file=sys.stderr, flush=True)
            raise


if __name__ == '__main__':
    main()

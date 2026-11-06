"""Install verified staged batch results; rollback preserves the three earlier repairs."""
import argparse
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import stat
import sys
from types import CodeType

COMMON_PATH = Path(__file__).resolve().parent.parent / 'fixed_card_patch_20261102/install.py'
spec = importlib.util.spec_from_file_location('fixed_card_install_common', COMMON_PATH)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

ORIGINAL = 'ea0cf8245dbb97d4b2a2ea2ff63af89dc0ab81ffd99b7a9d3de4aab84cc7e34c'
FIXED = '6889ba8a4a722455fa7d07633fe7765e3f61d56f48e4c584e0aee7977db17b32'


def install(root, execute=False, rollback=False):
    target = root / 'model_design/muse_encoder/pretrain/parallel_dashboard.py'
    package = root / 'tools/retry_commit_patch_20261103'
    backup = package / 'installed_backup/parallel_dashboard.py'
    if target.is_symlink():
        raise RuntimeError('Production is a symlink; stopped')
    before, after = (FIXED, ORIGINAL) if rollback else (ORIGINAL, FIXED)
    replacement = common.checked(backup if rollback else package / 'candidate/retry_parallel_dashboard.py', after)
    current = common.checked(target, before)
    compile(replacement, str(target), 'exec')
    if backup.exists():
        common.checked(backup, ORIGINAL)
    print(f'CHECK PASS: {"rollback" if rollback else "install"}; target={target}', flush=True)
    if not execute:
        print('CHECK ONLY: production unchanged; use --execute to apply.', flush=True)
        return target
    if not rollback and not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        with backup.open('xb') as stream:
            stream.write(current)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(backup, stat.S_IMODE(target.stat().st_mode))
        common.checked(backup, ORIGINAL)
    print(f'BACKUP: {backup}', flush=True)
    common.replace_checked(target, before, replacement, after)
    print(f'{"RESTORED" if rollback else "INSTALLED"}: {target}', flush=True)
    print(f'SHA256: {after}', flush=True)
    return target


def verify_import(root, target, rollback):
    sys.dont_write_bytecode = True
    os.chdir(root)
    sys.path.insert(0, str(root / 'model_design'))
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    module = importlib.import_module('muse_encoder.pretrain.parallel_dashboard')
    if Path(module.__file__).resolve() != target.resolve():
        raise RuntimeError(f'Unexpected import: {module.__file__}')
    common.checked(target, ORIGINAL if rollback else FIXED)
    if ('devices' not in inspect.signature(module._parallel_map).parameters or
            not callable(getattr(module, '_init_encoding_worker', None)) or
            'central_payload' not in module.encode_records_parallel.__code__.co_varnames or
            not hasattr(module._encode_items, '__wrapped__')):
        raise RuntimeError('A preceding fixed-card/cache/no_grad repair is missing')
    chunks = [c for c in inspect.unwrap(module._encode_items).__code__.co_consts
              if isinstance(c, CodeType) and c.co_name == 'encode_chunk']
    pending = {'pending_ids', 'pending_embeddings', 'pending_tokens'}
    if len(chunks) != 1 or (pending <= set(chunks[0].co_varnames)) != (not rollback):
        raise RuntimeError('Loaded retry implementation differs from expected version')
    print(f'IMPORT PASS: {module.__file__}; preceding three repairs retained; no training started', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=common.DEFAULT_ROOT)
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

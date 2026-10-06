"""Verify and unpack the published project; never install packages or run a simulation."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'IsaacLab_RS_final'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(destination, files):
    for name, digest in files.items():
        path = destination / name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError(f'File missing or changed; nothing overwritten: {path}')
    if not (destination / 'isaaclab.sh').stat().st_mode & 0o111:
        raise ValueError('isaaclab.sh is not executable; check the destination permissions.')


def prepare(destination):
    metadata = json.loads((ROOT / 'artifacts/project/manifest.json').read_text())
    archive = ROOT / 'artifacts/project' / metadata['archive']
    if sha(archive) != metadata['sha256']:
        raise ValueError('Archive SHA-256 mismatch. Download the complete repository again.')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        expected = metadata['files']
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or len(path.parts) < 2
                    or path.parts[0] != PREFIX or not member.isfile()):
                raise ValueError(f'Unexpected archive entry: {member.name}')
            name = str(path.relative_to(PREFIX))
            if name in names or name not in expected:
                raise ValueError(f'Unexpected or duplicate file: {name}')
            if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != expected[name]:
                raise ValueError(f'Archived file SHA-256 mismatch: {name}')
            names.add(name)
        if names != set(expected) or len(members) != metadata['file_count']:
            raise ValueError('Archive inventory mismatch.')
        # Existing files must match exactly; never overwrite a user's checkout.
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError(f'Destination is not a regular directory: {destination}')
            verify_files(destination, expected)
            print(f'[OK] Existing project verified: {destination}')
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.ant-unpack-', dir=destination.parent) as temporary:
            staging = Path(temporary) / PREFIX
            staging.mkdir()
            for member in members:
                relative = PurePosixPath(member.name).relative_to(PREFIX)
                target = staging / str(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
            verify_files(staging, expected)
            # mkdir ensures even a concurrent new destination is not overwritten.
            destination.mkdir()
            for path in staging.iterdir():
                shutil.move(str(path), str(destination / path.name))
    print(f'[OK] Extracted and verified {len(expected)} files: {destination}')
    print('[NEXT] Activate the Isaac Sim conda environment and install this checkout: ./isaaclab.sh -i rsl_rl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path.home() / PREFIX)
    args = parser.parse_args()
    try:
        prepare(args.destination.expanduser().absolute())
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, f'[ERROR] {error}\n')

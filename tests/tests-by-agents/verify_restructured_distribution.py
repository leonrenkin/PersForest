"""Check built archives against src, then safely unpack an sdist for rebuilding.

Usage
-----
    uv run python tests/tests-by-agents/verify_restructured_distribution.py DIST_DIR

Archives must already exist. This helper does not download, build, or install
anything. It records distribution members and SHA256 digests beside the archives.
"""
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import tarfile
import tomllib
import zipfile


def main() -> None:
    """Verify that wheel/sdist ship exactly the src package's Python files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dist_dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    dist_info = f"persforest-{project['version']}.dist-info/"
    expected = {path.relative_to(root / 'src').as_posix(): path.read_bytes()
                for path in (root / 'src/persforest').rglob('*.py')}
    wheel, = args.dist_dir.glob('*.whl')
    sdist, = args.dist_dir.glob('*.tar.gz')
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        actual = {item.filename: archive.read(item) for item in archive.infolist()
                  if item.filename.startswith('persforest/') and not item.is_dir()}
        assert actual == expected, ('wheel package mismatch', actual.keys() ^ expected.keys())
        assert all(name.startswith(('persforest/', dist_info)) for name in members)
        metadata = archive.read(dist_info + 'METADATA').decode()
        constraint = Parser().parsestr(metadata)['Requires-Python']
        assert {part.strip() for part in constraint.split(',')} == {
            part.strip() for part in project['requires-python'].split(',')}, constraint
        assert any(name.endswith('/LICENSE') for name in members)
        wheel_members = members
    with tarfile.open(sdist) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        prefix = names[0].split('/')[0] + '/'
        allowed = {'LICENSE', 'README.md', 'pyproject.toml', 'pyproject.toml.orig', 'PKG-INFO'}
        assert all(name.removeprefix(prefix).startswith('src/persforest/')
                   or name.removeprefix(prefix) in allowed for name in names), names
        actual = {name.removeprefix(prefix + 'src/'): archive.extractfile(name).read()
                  for name in names if name.startswith(prefix + 'src/persforest/')}
        assert actual == expected, ('sdist package mismatch', actual.keys() ^ expected.keys())
        unpacked = args.dist_dir / 'unpacked'
        unpacked.mkdir(exist_ok=True)
        archive.extractall(unpacked, filter='data')
    report = dict(package_files=len(expected), wheel_members=wheel_members, sdist_members=names,
                  hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [wheel, sdist]})
    (args.dist_dir / 'verified-contents.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'VERIFIED: {len(expected)} src files in both archives; no reference/private/generated files.')
    print('Unpacked:', unpacked / prefix.rstrip('/'))


if __name__ == '__main__':
    main()

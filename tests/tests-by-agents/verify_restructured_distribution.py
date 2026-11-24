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
    """Verify the archives: Python files of src, the compiled engine, and its sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dist_dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    dist_info = f"persforest-{project['version']}.dist-info/"
    expected = {path.relative_to(root / 'src').as_posix(): path.read_bytes()
                for path in (root / 'src/persforest').rglob('*.py')}
    # Sources and build files of the compiled engine, including vendored
    # third-party code with its license, shipped in the sdist only.
    third_party = sorted(path for path in (root / 'cpp/third_party').rglob('*') if path.is_file())
    engine_sources = {path.relative_to(root).as_posix(): path.read_bytes()
                      for path in [root / 'CMakeLists.txt', *sorted((root / 'cpp').glob('*.[ch]pp')),
                                   *third_party]}
    wheel, = args.dist_dir.glob('*.whl')
    sdist, = args.dist_dir.glob('*.tar.gz')
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        actual = {item.filename: archive.read(item) for item in archive.infolist()
                  if item.filename.startswith('persforest/') and not item.is_dir()}
        extensions = [name for name in actual if name.startswith('persforest/_engine.')
                      and name.endswith(('.so', '.pyd'))]
        assert len(extensions) == 1, ('expected one compiled engine module', extensions)
        actual.pop(extensions[0])
        assert actual == expected, ('wheel package mismatch', actual.keys() ^ expected.keys())
        assert all(name.startswith(('persforest/', dist_info)) for name in members)
        metadata = archive.read(dist_info + 'METADATA').decode()
        constraint = Parser().parsestr(metadata)['Requires-Python']
        assert {part.strip() for part in constraint.split(',')} == {
            part.strip() for part in project['requires-python'].split(',')}, constraint
        # Every license in `license-files` ships with the wheel: the compiled
        # engine contains the vendored BSD-3-Clause code.
        for license_file in project['license-files']:
            shipped = archive.read(f'{dist_info}licenses/{license_file}')
            assert shipped == (root / license_file).read_bytes(), ('wheel license mismatch', license_file)
        wheel_members = members
    with tarfile.open(sdist) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        prefix = names[0].split('/')[0] + '/'
        allowed = {'LICENSE', 'README.md', 'pyproject.toml', 'pyproject.toml.orig', 'PKG-INFO', *engine_sources}
        assert all(name.removeprefix(prefix).startswith('src/persforest/')
                   or name.removeprefix(prefix) in allowed for name in names), names
        shipped_sources = {name.removeprefix(prefix): archive.extractfile(name).read()
                           for name in names if name.removeprefix(prefix) in engine_sources}
        assert shipped_sources == engine_sources, ('sdist engine sources mismatch',
                                                   shipped_sources.keys() ^ engine_sources.keys())
        actual = {name.removeprefix(prefix + 'src/'): archive.extractfile(name).read()
                  for name in names if name.startswith(prefix + 'src/persforest/')}
        assert actual == expected, ('sdist package mismatch', actual.keys() ^ expected.keys())
        unpacked = args.dist_dir / 'unpacked'
        unpacked.mkdir(exist_ok=True)
        archive.extractall(unpacked, filter='data')
    report = dict(package_files=len(expected), wheel_members=wheel_members, sdist_members=names,
                  hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [wheel, sdist]})
    (args.dist_dir / 'verified-contents.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'VERIFIED: {len(expected)} src files in both archives, the compiled engine in the wheel and its '
          f'{len(engine_sources)} source files in the sdist; no reference/private/generated files.')
    print('Unpacked:', unpacked / prefix.rstrip('/'))


if __name__ == '__main__':
    main()

import os
import subprocess

from setuptools import setup

MSA_SAIL_VERSION = '0.1.1'  # msa_for_sail version
RELEASE_VERSION = 'v0.1.0'  # Release version

this_dir = os.path.dirname(os.path.abspath(__file__))
ppu_sdk = os.environ.get('PPU_SDK') or os.environ.get('PPU_HOME') or '/usr/local/PPU_SDK'


def get_ppu_sdk_version():
    # Query version macros using the compiler belonging to the selected SDK.
    try:
        compiler = os.path.join(ppu_sdk, 'bin', 'hgcc')
        output = subprocess.check_output(
            [compiler, '-E', '-x', 'cu', '/dev/stdin', '-o', '/dev/stdout'],
            input=(
                'PPU_VERSION __HGGCCC_VER_MAJOR__ '
                '__HGGCCC_VER_MINOR__ __HGGCCC_VER_BUILD__\n'
            ),
            text=True,
            timeout=10,
        )
        major, minor, build = map(int, output.rsplit('PPU_VERSION', 1)[1].split())
        return (major, minor, build)
    except Exception:
        pass
    return (0, 0, 0)


def get_package_version(ppu_sdk_version):
    try:
        from setuptools_scm import Configuration
        from setuptools_scm.git import parse
    except ImportError as error:
        raise RuntimeError('Missing build dependency: install setuptools-scm first') from error

    release_tag = f'{MSA_SAIL_VERSION}+{RELEASE_VERSION}'
    # Match only this release tag; uncommitted changes never affect the version.
    scm = parse(this_dir, Configuration(), describe_command=[
        'git', 'describe', '--tags', '--long', '--match', release_tag,
    ])
    if scm is None or scm.node is None:
        raise RuntimeError('Git commit information unavailable; build from a Git repository with commit history')
    if ppu_sdk_version == (0, 0, 0):
        raise RuntimeError('PPU SDK version unavailable; check hgcc --version for the selected SDK')

    ppu_version = '.'.join(map(str, ppu_sdk_version))
    local_version = f'{RELEASE_VERSION}.ppu{ppu_version}'
    if scm.distance in (None, 0):
        return f'{MSA_SAIL_VERSION}+{local_version}'
    return f'{MSA_SAIL_VERSION}.dev{scm.distance}+{local_version}.{scm.node[:9]}'


package_version = get_package_version(get_ppu_sdk_version())

setup(version=package_version)

from setuptools import setup
import os
from pathlib import Path

scripts_dir = 'feFlow/scripts'

scripts = [str(f) for f in Path(scripts_dir).glob('*') if f.is_file()]

bin_dir = os.path.expanduser('~/.local/bin')


setup(
    name='feFlow',
    version='1.0',
    author='Chayut Teeraratkul',
    author_email='chayut.teeraratkul@colorado.edu',
    packages=['feFlow'],
    url='',
    license='See LICENSE.txt',
    description='FEniCS for flow physics problems',
    long_description=open('README.md').read(),
    scripts=scripts,
    data_files=[(bin_dir, scripts)],
)

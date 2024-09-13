from setuptools import setup
import os
from pathlib import Path

# Path to your Bash scripts
scripts_dir = Path('src/flatiron_tk/scripts')
scripts = [str(f) for f in scripts_dir.glob("*")]

setup(
    name='flatiron_tk',
    version='1.0.0',
    author='Chayut Teeraratkul',
    author_email='chayut.teeraratkul@colorado.edu',
    packages=['flatiron_tk'],
    package_dir={'': 'src/'},
    url='',
    license='See LICENSE.txt',
    description='FLow And Transport Finite element',
    long_description=open('README.md').read(),
    scripts=scripts,
)

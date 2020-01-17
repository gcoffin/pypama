import pathlib
from distutils.core import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.rst").read_text()

setup(name='pypama',
      version='1.0',
      description='Python Pattern Matching',
      long_description=README,
      author='Guillaume Coffin',
      author_email='guill.coffin@gmail.com',
      license='GPLv3',
      url='http://github.com/gcoffin/pypama',
      packages=['pypama'],
      )

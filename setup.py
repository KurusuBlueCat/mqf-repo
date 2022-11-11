from setuptools import setup

import re
VERSIONFILE = "mqf_toolbox/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='mqf_toolbox',
      version=verstr,
      description='Useful Stuff for SMU MQF',
      author='Miti Nopnirapath',
      author_email='mitip.2022@mqf.smu.edu.sg',
      install_requires=["setuptools",
                        ],
      packages=['mqf_toolbox',
                'mqf_toolbox.asset_pricing'],
      )

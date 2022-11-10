from setuptools import setup

setup(name='mqf_toolbox',
      version='0.1.1',
      description='Useful Stuff for SMU MQF',
      author='Miti Nopnirapath',
      author_email='mitip.2022@mqf.smu.edu.sg',
      install_requires=["numpy",
                        "pandas",
                        "statsmodels",
                        "matplotlib",
                        "setuptools",
                        ],
      packages=['mqf_toolbox', 'mqf_toolbox.asset_pricing'],
      )

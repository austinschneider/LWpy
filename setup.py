from setuptools import setup

setup(name='LWpy',
      version='0.0.0',
      description='Module for weighting LeptonInjector simulation',
      url='https://github.com/austinschneider/LWpy',
      author='Austin Schneider',
      author_email='physics.schneider@gmail.com',
      license='L-GPL-2.1',
      packages=['LWpy', 'LWpy/generator', 'LWpy/resources', 'LWpy/tests'],
      package_data={'LWpy': [
          'resources/crosssections/csms_differential_v1.0/*.fits',
          'resources/earthparams/materials/*.dat',
          'resources/earthparams/densities/*.dat',
          'resources/interactions/*.py',
          ]},
      include_package_data=True,
      zip_safe=False)

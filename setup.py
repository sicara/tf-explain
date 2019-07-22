from setuptools import setup
from setuptools import find_packages


setup(name='Mentat',
      version='0.0.1',
      description='Interpretability Callbacks for Tensorflow 2.0',
      author='Raphaël Meudec',
      author_email='raphaelm@sicara.com',
      url='https://github.com/sicara/mentat',
      license='MIT',
      install_requires=['opencv-python>=4.1.0.25',
                        'pillow>=6.1.0'],
      extras_require={
          'tests': ['pylint>=2.3.1',
                    'pytest>=5.0.1',
                    'pytest-timeout>=1.3.3',
                    'pytest-mock>=1.10.4',
                    'pytest-cov>=2.7.1',
                    'tox>=3.13.2'],
          'examples': ['opencv-python'],
      },
      packages=find_packages()
)

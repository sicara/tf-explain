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
          'tests': ['pylint',
                    'pytest',
                    'pytest-mock',
                    'pytest-cov'],
          'examples': ['opencv-python'],
      },
      packages=find_packages()
)

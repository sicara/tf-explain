from setuptools import setup
from setuptools import find_packages


setup(name='tf-explain',
      version='0.0.1',
      description='Interpretability Callbacks for Tensorflow 2.0',
      author='Raphaël Meudec',
      author_email='raphaelm@sicara.com',
      url='https://github.com/sicara/tf-explain',
      license='MIT',
      install_requires=['opencv-python>=4.1.0.25',],
      extras_require={
          'tests': ['black>=19.3b0',
                    'pylint>=2.3.1',
                    'pytest>=5.0.1',
                    'pytest-timeout>=1.3.3',
                    'pytest-mock>=1.10.4',
                    'pytest-cov>=2.7.1',
                    'tox>=3.13.2'],
      },
      packages=find_packages()
)

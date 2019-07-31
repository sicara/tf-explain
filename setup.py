from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as f:
    long_description = f.read()


setup(name='tf-explain',
      version='0.0.2-alpha',
      description='Interpretability Callbacks for Tensorflow 2.0',
      long_description=long_description,
      long_description_content_type="text/markdown",
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
          'publish': ['bumpversion>=0.5.3',
                      'twine>=1.13.0']
      },
      packages=find_packages()
)

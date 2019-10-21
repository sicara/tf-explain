Contributing
############

Contributions are welcome on this repo! Follow this guide to see how you can help.

What can I do?
**************

There are multiple ways to give a hand on this repo:

* resolve issues already opened
* tackle new features from the roadmap
* fix typos, improve code quality, code coverage

Guidelines
**********

Tests
^^^^^

tf-explain is run against Python 3.6 and 3.7, for Tensorflow beta. We use
`tox <https://github.com/pytest-dev/pytest>`_ (available with :code:`pip`) to perform the tests.
All the submitted code should be unit tested (we use `pytest <https://github.com/pytest-dev/pytest>`_).

To run all the tests, install required packages with :code:`pip install -e .[tests]` and then run :code:`tox` in a
terminal.

Code Format
^^^^^^^^^^^

All code is formatted with `Black <https://www.github.com/psf/black>`_ (available with :code:`pip`). When opening your PR,
make sure your code is formatted or Travis will fail. To format your code, simply call :code:`make black`.

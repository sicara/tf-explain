Overview
########


Installation
************

tf-explain is not available yet on Pypi. To install it, clone it locally:
::
    git clone https://www.github.com/sicara/tf-explain
    pip install -e .


Tensorflow Compatibility
************************

tf-explain is compatible with Tensorflow 2. It is not declared as a dependency
to let you choose between CPU and GPU versions. Additionally to the previous install,
run::
    # For CPU version
    pip install tensorflow==2.0.0-beta1
    # For GPU version
    pip install tensorflow-gpu==2.0.0-beta1

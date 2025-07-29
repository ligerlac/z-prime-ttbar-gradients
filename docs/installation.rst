Installation
============

The framework can be installed using either ``conda`` or ``pip``.

Conda (Recommended)
-------------------
.. code-block:: bash

    conda env create -f environment.yml
    conda activate zprime_diff_analysis

Pip
---
.. code-block:: bash

    conda create -n zprime_diff_analysis python=3.10
    conda activate zprime_diff_analysis
    pip install -r requirements.txt

Virtual Environment (Alternative)
---------------------------------
.. code-block:: bash

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

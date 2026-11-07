PY ?= python

.PHONY: install test lint verify equivalence quickstart anonymity check

install:      ## install the package with the test and lint tools
	$(PY) -m pip install -e ".[dev]"

test:         ## run the test suite
	$(PY) -m pytest

lint:         ## static checks
	ruff check src scripts tests examples

verify:       ## recompute the registration hashes and the B27 record
	$(PY) scripts/verify_prereg.py

equivalence:  ## compare the package with the final notebook (notebooks/EXP9/EXP9_v8_final.ipynb)
	$(PY) scripts/check_notebook_equivalence.py --notebook notebooks/EXP9/EXP9_v8_final.ipynb

quickstart:   ## identification and re-degradation on synthetic tiles
	$(PY) examples/quickstart.py

anonymity:    ## search the repository for author-identifying content
	$(PY) scripts/check_anonymity.py

check: lint test verify anonymity

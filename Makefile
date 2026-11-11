.PHONY: figures figures-check test check

GENS = tools/render_hero.py tools/render_map.py tools/render_panel.py

figures:
	for g in $(GENS); do python3 $$g --write || exit 1; done

figures-check:
	for g in $(GENS); do python3 $$g --check || exit 1; done

test:
	python3 -m pytest -q

check: figures-check
	python3 tools/check_replay.py
	python3 tools/check_readme.py

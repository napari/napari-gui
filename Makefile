.PHONY: docs watch

docs:
	rm -rf docs/_build/
	find docs/api ! -name 'index.rst' -type f -exec rm -f {} +
	pip install -qr docs/requirements.txt
	jb build docs

typestubs:
	python -m napari.utils.stubgen

# If the first argument is "watch"...
ifeq (watch,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "watch"
  WATCH_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(WATCH_ARGS):;@:)
endif

# examples:
# make watch ~/Desktop/Untitled.png
# make watch -- -w animation  # -- is required for passing flags to napari

watch:
	@echo "running: napari $(WATCH_ARGS)"
	@echo "Save any file to restart napari\nCtrl-C to stop..\n" && \
		watchmedo auto-restart -R \
			--ignore-patterns="*.pyc*" -D \
			--signal SIGKILL \
			napari -- $(WATCH_ARGS) || \
		echo "please run 'pip install watchdog[watchmedo]'"

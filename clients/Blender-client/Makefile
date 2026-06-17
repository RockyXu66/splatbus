SPLATBUS_SRC := ../../splatbus
WHEELS     := ./wheels
BLENDER    ?= blender

# Build the splatbus Python package and copy the resulting wheel into ./wheels/
.PHONY: splatbus-wheel
splatbus-wheel:
	uv build --directory $(SPLATBUS_SRC)
	cp $(SPLATBUS_SRC)/dist/splatbus-*.whl $(WHEELS)/

# Download splatbus' dependency wheels for all supported platforms
.PHONY: pull-deps
pull-deps:
	python3 sync_wheels.py

# Remove all generated build/dist artifacts, and only delete the built splatbus wheel from wheels
.PHONY: clean
clean:
	rm -rf ./build ./dist
	rm -f $(WHEELS)/splatbus-*.whl

# Pull deps, build splatbus wheel, then package the Blender extension
.PHONY: build
build: pull-deps
	python3 build_addon.py "$(BLENDER)"

# Pull deps, build splatbus wheel, then package the Blender extension
.PHONY: build-linux
build-linux: pull-deps
	python3 build_addon.py "$(BLENDER)" --platform linux-x64

# Detect the active Blender version for the user extensions path.
BLENDER_VERSION := $(shell $(BLENDER) --version 2>/dev/null | grep -m1 -oE '[0-9]+\.[0-9]+')
EXTENSIONS_DIR := $(HOME)/.config/blender/$(BLENDER_VERSION)/extensions/user_default/splatbus

# Full extension install via Blender's extension manager.
.PHONY: install
install: build
	$(BLENDER) --background --command extension install-file dist/splatbus-0.0.1.zip -r user_default -e

# Fast development install: only copy the addon source file.
# Use this when only src/__init__.py has changed and the manifest/wheels are unchanged.
.PHONY: install-dev
install-dev: build
	@mkdir -p "$(EXTENSIONS_DIR)"
	@cp src/__init__.py "$(EXTENSIONS_DIR)/__init__.py"
	@echo "Copied src/__init__.py to $(EXTENSIONS_DIR)"

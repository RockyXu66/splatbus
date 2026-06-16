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

# Convenience wrapper around build.py — the manifest lives there.

.PHONY: build verify clean
build:
	python3 build.py build

verify:
	python3 build.py verify

clean:
	python3 build.py clean

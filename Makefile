# Convenience wrapper. The manifest and verifier live in verify.py.

.PHONY: verify dashboard
verify:
	python3 verify.py

dashboard:
	python3 scripts/build_dashboard.py

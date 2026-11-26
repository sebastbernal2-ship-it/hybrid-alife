PYTHON ?= python
RUN_DIR ?= outputs/runs/smoke200
QUICK_RUN_DIR ?= outputs/runs/base

.PHONY: help gallery gallery-quick gallery-report gallery-clean ablation-gallery test

help:
	@echo "Common targets:"
	@echo "  make gallery        - smoke200 run + report + plots (~110 s CPU)"
	@echo "  make gallery-quick  - base.yaml run + plots (seconds, 2 generations)"
	@echo "  make gallery-report - regenerate report+plots from existing $(RUN_DIR)"
	@echo "  make ablation-gallery - run the 4-config ablation matrix"
	@echo "  make gallery-clean  - remove $(RUN_DIR)/plots/"
	@echo "  make test           - pytest -q"
	@echo ""
	@echo "See docs/GALLERY.md for what each plot means."

gallery:
	$(PYTHON) scripts/run_sim.py --config configs/smoke200.yaml
	$(PYTHON) scripts/generate_report.py $(RUN_DIR)
	@echo "Plots: $(RUN_DIR)/plots/  Report: $(RUN_DIR)/report.md"

gallery-quick:
	$(PYTHON) scripts/run_sim.py --config configs/base.yaml
	$(PYTHON) scripts/generate_report.py $(QUICK_RUN_DIR)
	@echo "Plots: $(QUICK_RUN_DIR)/plots/  Report: $(QUICK_RUN_DIR)/report.md"

gallery-report:
	$(PYTHON) scripts/generate_report.py $(RUN_DIR)

ablation-gallery:
	$(PYTHON) scripts/run_ablation_matrix.py \
	    --configs configs/smoke200.yaml configs/ablation_no_comms.yaml \
	              configs/ablation_static_world.yaml configs/ablation_uniform_field.yaml \
	    --seeds 3 --include-shadow \
	    --out-dir outputs/ablation_matrix
	@echo "Summary: outputs/ablation_matrix/summary.md"

gallery-clean:
	rm -rf $(RUN_DIR)/plots

test:
	pytest -q

SAN_FLAGS := -O1 -g -std=c++17 -fPIC -Wall -Wextra -fno-omit-frame-pointer \
	-fsanitize=address,undefined -fno-sanitize-recover=all
# Python is not instrumented, so the runtime must be preloaded. libstdc++ goes with it so
# ASan can intercept C++ throws from jaxlib. Leak checking is off: CPython "leaks" by design.
SAN_ENV = LD_PRELOAD="$$(g++ -print-file-name=libasan.so) $$(g++ -print-file-name=libstdc++.so)" \
	ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 PYTHONPATH=src
# Deliberately undersized buffers prove that the sanitizer is really active.
SAN_CANARY := import numpy as np; from hybrid_alife.physics import lbm; \
	a = np.zeros(1000); lbm._load().lbm_init(64, 64, 1.0, a, a, a)

# Replaces the normal native libraries; restore them with `make -B -C native`.
sanitize-build:
	$(MAKE) -B -C native CXXFLAGS="$(SAN_FLAGS)"

# -s keeps a sanitizer report visible: it aborts the process before pytest can print captures.
sanitize-test:
	$(SAN_ENV) $(PYTHON) -c "$(SAN_CANARY)" 2>&1 | grep -q "AddressSanitizer: heap-buffer-overflow"
	$(SAN_ENV) $(PYTHON) -m pytest -q -s tests/test_avida_*.py tests/test_lbm_*.py \
		tests/test_immersed_boundary.py

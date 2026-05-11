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

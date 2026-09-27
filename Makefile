# dt-viz — visualizations of the Distribution Transformer models trained in odin-ml,
# drawn with odin-ui-v2. Both libraries are pinned clones in deps/ (never your working
# copies); point ML_DIR / UI_DIR at a local checkout to develop against it instead.
ODIN    ?= odin
FAST    := -o:speed -no-bounds-check
BUILD   := build

ML_REPO := https://github.com/joekerenski/odin-ml.git
ML_REV  := fec6cb905ac179a2029133c9d5d42b35de995aef
ML_DIR  ?= deps/odin-ml
UI_REPO := https://github.com/joekerenski/odin-ui-v2.git
UI_REV  := f91b568bf84afe1d9bc9451e745ae7248d122498
UI_DIR  ?= deps/odin-ui-v2

# The trained M4 model. odin-ml keeps weights out of git, so train it once with
# `make model` (in the pinned odin-ml clone) or pass MODEL=path/to/fusion_dt4.safetensors.
MODEL   ?= models/fusion_dt4.safetensors

COLLECTIONS := -collection:odinml=$(ML_DIR) -collection:odinui=$(UI_DIR)

.PHONY: run build deps model clean

run: build
	@test -f $(MODEL) || { echo "no model at $(MODEL): run 'make model' (trains M4 in odin-ml, ~40 min on an M-series GPU) or pass MODEL=path"; exit 1; }
	$(BUILD)/radar $(MODEL)

build: deps
	$(ODIN) build src $(FAST) $(COLLECTIONS) -out:$(BUILD)/radar

# Clone on first use; afterwards make sure the checkout is at the pinned revision.
# Directories passed in from outside (ML_DIR=../odin-ml) are left alone.
pin = test -d $(2)/.git || git clone -q $(1) $(2); \
	if [ "$$(git -C $(2) rev-parse HEAD)" != "$(3)" ]; then \
		git -C $(2) cat-file -e $(3)^{commit} 2>/dev/null || git -C $(2) fetch -q; \
		git -C $(2) checkout -q $(3) && echo "$(2) -> $(3)"; \
	fi

deps:
ifeq ($(ML_DIR),deps/odin-ml)
	@$(call pin,$(ML_REPO),$(ML_DIR),$(ML_REV))
endif
ifeq ($(UI_DIR),deps/odin-ui-v2)
	@$(call pin,$(UI_REPO),$(UI_DIR),$(UI_REV))
endif

# Train the fusion model with the pinned odin-ml (Metal if available) and keep a copy here.
model: deps
	cd $(ML_DIR) && ML_DEVICE=metal $(ODIN) run dt/fusion $(FAST) -out:fusion -- train
	mkdir -p $(dir $(MODEL)) && cp $(ML_DIR)/models/fusion_dt4.safetensors $(MODEL)

clean: ; rm -rf $(BUILD)

$(shell mkdir -p $(BUILD))

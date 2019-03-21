

FPS1 := $(wildcard 1fps/*.png)

WHITEOUT_DIR := $(shell pwd)/whiteout
SWIR_DIR := $(shell pwd)/false_swir

whiteout: $(FPS1)
	../analysis/make_products.py --whiteout --log DEBUG --output-dir $(WHITEOUT_DIR) $^

false_swir: $(FPS1)
	../analysis/make_products.py --false-swir --log DEBUG --output-dir $(SWIR_DIR) $^


single_band_1fps: $(FPS1)
	../analysis/single_band_tracking.py --log DEBUG --output-dir $(shell pwd)/$@ $^

single_band_whiteout: $(wildcard whiteout/*.png)
	../analysis/single_band_tracking.py --log DEBUG --output-dir $(shell pwd)/$@ $^



.PHONY: whiteout false_swir single_band_1fps single_band_whiteout




FULL := $(wildcard full/*.png)
FPS1 := $(wildcard 1fps/*.png)




##

FULL_WHITEOUT_DIR := $(shell pwd)/full_products/whiteout

full/whiteout: $(FULL)
	../analysis/make_products.py --whiteout --log DEBUG --output-dir $(FULL_WHITEOUT_DIR) $^

full/whiteout_movie: $(FULL_WHITEOUT_DIR)
	ffmpeg -r 30 -i $(FULL_WHITEOUT_DIR)/frame-%05d.png -vcodec h264 -profile high -pix_fmt yuv420p -b:v 16000k -y full_whiteout.mov



## Deal with 1FPS video

FPS_WHITEOUT_DIR := $(shell pwd)/1fps_products/whiteout
FPS_SWIR_DIR     := $(shell pwd)/1fps_products/false_swir

1fps/whiteout: $(FPS1)
	../analysis/make_products.py --whiteout --log DEBUG --output-dir $(FPS_WHITEOUT_DIR) $^

1fps/false_swir: $(FPS1)
	../analysis/make_products.py --false-swir --log DEBUG --output-dir $(FPS_SWIR_DIR) $^

single_band_1fps: $(FPS1)
	../analysis/single_band_tracking.py --log DEBUG --output-dir $(shell pwd)/$@ $^

single_band_whiteout: $(wildcard $(FPS_WHITEOUT_DIR)/*.png)
	../analysis/single_band_tracking.py --log DEBUG --output-dir $(shell pwd)/$@ $^



.PHONY: whiteout false_swir single_band_1fps single_band_whiteout

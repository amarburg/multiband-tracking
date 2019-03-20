

FPS1 := $(wildcard 1fps/*.png)

WHITEOUT := $(subst 1fps,whiteout,$(FPS1))
WHITEOUT_DIR := $(shell pwd)/whiteout

whiteout: $(FPS1)
	../analysis/make_whiteout.py --log DEBUG --output-dir $(WHITEOUT_DIR) $^

.PHONY: whiteout

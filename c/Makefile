# Build the C inference reference.
#
#   make        compile kws_infer (needs model_data.h + sample.h, see below)
#   make run    build and run it
#   make clean
#
# model_data.h and sample.h are generated from a trained model by:
#   python ../scripts/export_c.py
# A pre-generated pair is committed so this builds out of the box.

CC     ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -O2

kws_infer: kws_infer.c model_data.h sample.h
	$(CC) $(CFLAGS) kws_infer.c -o kws_infer -lm

.PHONY: run clean
run: kws_infer
	./kws_infer

clean:
	rm -f kws_infer

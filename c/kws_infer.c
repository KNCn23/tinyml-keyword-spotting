/* kws_infer - run the quantised keyword model in pure C.
 *
 * This is the "deployment" side of the project: no Python, no floating-point
 * matrix library, just the int8 weights baked into model_data.h and a single
 * feature vector from sample.h. It reproduces, byte for byte, the integer math
 * in kws/quantize.py, which is what lets you trust that what you trained is what
 * runs on the device.
 *
 * Build and run:   make            (in this directory)
 *                  ./kws_infer
 *
 * The arrays in model_data.h and sample.h are produced by:
 *     python scripts/export_c.py
 */
#include <stdio.h>
#include <math.h>

#include "model_data.h"
#include "sample.h"

/* Dynamic per-vector symmetric quantisation: x -> int8 q, returns the scale.
 * Mirrors kws.quantize._quantize_vec. */
static float quantize_vec(const float *x, int n, signed char *q) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > max_abs)
            max_abs = a;
    }
    float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    for (int i = 0; i < n; i++) {
        float v = roundf(x[i] / scale);
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        q[i] = (signed char)v;
    }
    return scale;
}

/* One quantised affine layer: int32 accumulation, then dequantise + bias.
 * W is row-major [in][out]. */
static void affine(const signed char *xq, float sx, const signed char *W,
                   float sw, const float *bias, int in, int out, float *y) {
    for (int o = 0; o < out; o++) {
        long acc = 0;                       /* int32-range accumulator */
        for (int i = 0; i < in; i++)
            acc += (long)xq[i] * (long)W[i * out + o];
        y[o] = (float)acc * (sx * sw) + bias[o];
    }
}

int main(void) {
    /* Layer 1: input -> hidden, then ReLU. */
    signed char xq[KWS_IN_DIM];
    float sx = quantize_vec(KWS_SAMPLE, KWS_IN_DIM, xq);

    float h[KWS_HIDDEN];
    affine(xq, sx, KWS_W1, KWS_S1, KWS_B1, KWS_IN_DIM, KWS_HIDDEN, h);
    for (int i = 0; i < KWS_HIDDEN; i++)
        if (h[i] < 0.0f)
            h[i] = 0.0f;                    /* ReLU */

    /* Layer 2: hidden -> classes. */
    signed char hq[KWS_HIDDEN];
    float sh = quantize_vec(h, KWS_HIDDEN, hq);

    float logits[KWS_N_CLASSES];
    affine(hq, sh, KWS_W2, KWS_S2, KWS_B2, KWS_HIDDEN, KWS_N_CLASSES, logits);

    /* Argmax over the class logits. */
    int best = 0;
    for (int c = 1; c < KWS_N_CLASSES; c++)
        if (logits[c] > logits[best])
            best = c;

    printf("class logits:\n");
    for (int c = 0; c < KWS_N_CLASSES; c++)
        printf("  %-10s % .4f\n", KWS_LABELS[c], logits[c]);

    printf("\npredicted: %s\n", KWS_LABELS[best]);
    printf("expected : %s\n", KWS_LABELS[KWS_TRUE_LABEL]);
    printf("%s\n", best == KWS_TRUE_LABEL ? "MATCH" : "MISMATCH");

    return best == KWS_TRUE_LABEL ? 0 : 1;
}

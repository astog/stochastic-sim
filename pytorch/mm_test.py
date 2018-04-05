from __future__ import print_function
import torch
import numpy as np
import stoch_bin as sb


def approx_mm_int(t1, t2, passes):
    out = None
    for ipass in xrange(passes):
        t1s = 1 - sb.binarize(t1, bipolar=True).type(torch.IntTensor).mul_(2)
        t2s = 1 - sb.binarize(t2, bipolar=True).type(torch.IntTensor).mul_(2)
        if out is None:
            out = t1s.mm(t2s)
        else:
            out += t1s.mm(t2s)
    return out.type(type(t1)) / float(passes)


def approx_mm_int_alt(t1, t2, passes):
    t1s = passes - sb.to_samples(t1, passes, bipolar=True).mul_(2*passes)
    t2s = passes - sb.to_samples(t2, passes, bipolar=True).mul_(2*passes)
    return t1s.mm(t2s).type(type(t1))


def approx_mm_float(t1, t2, passes):
    out = None
    for ipass in xrange(passes):
        t1s = (0.5 - sb.binarize(t1, bipolar=True).type(type(t1))).sign_()
        t2s = (0.5 - sb.binarize(t2, bipolar=True).type(type(t2))).sign_()
        if out is None:
            out = t1s.mm(t2s)
        else:
            out += t1s.mm(t2s)
    return out / float(passes)

#!/usr/bin/env python3
# Copyright 2020 Deepwave Digital Inc.
""" Naive vs Precompiling CUDA kernels benchmark for cuSignal"""

import time
import cupy as cp
import numpy as np
import cusignal

buff_len = 2**19
n_test = 1000

# Create noise signal to simulate received data
noise = np.random.randn(buff_len) + 1j * np.random.randn(buff_len)
noise = noise.astype(np.complex64)

# Create shared memory buffer
buff = cusignal.get_shared_mem(buff_len, dtype=np.complex64)
buff[:] = noise

# Run benchmark cases

# No precompile of kernel
ti = time.monotonic()
for _ in range(n_test):
    buff[:] = noise
    sig_power1 = cp.power(cp.abs(buff), 2)
rate_msps = buff_len * n_test / (time.monotonic() - ti) / 1e6
print('Method 1: Data Rate = {:1.2f} MSPS'.format(rate_msps))

# Execute before loop to precompile kernel
sig_power2 = cp.power(cp.abs(buff), 2)
ti = time.monotonic()
for _ in range(n_test):
    buff[:] = noise
    sig_power2 = cp.power(cp.abs(buff), 2)
rate_msps = buff_len * n_test / (time.monotonic() - ti) / 1e6
print('Method 2: Data Rate = {:1.2f} MSPS'.format(rate_msps))


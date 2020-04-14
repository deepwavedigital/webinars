#!/usr/bin/env python3
""" PowerDetector benchmarking examples for cuSignal

On the AIR-T with AirStack 0.2 and cuSignal 0.14, this produces:
Data Rate = 155.09 MSPS
and utilizes 40 % of the GPU
"""
import time
import numpy as np
import cusignal
from powerdetector import PowerDetector

buff_len = 2**20
n_test = 10000
dec = 32
seg_len = 4096
threshold_db = 100

# Create noise signal to simulate received data
noise = np.random.randn(buff_len) + 1j * np.random.randn(buff_len)
noise = noise.astype(np.complex64)

# Create shared memory buffer
buff = cusignal.get_shared_mem(buff_len, dtype=np.complex64)
buff[:] = noise

detector = PowerDetector(buff, buff_len, dec, threshold_db)
t0 = time.monotonic()
for _ in range(n_test):
    buff[:] = noise
    output_segments = detector.detect(buff)
rate_msps = buff_len * n_test / (time.monotonic() - t0) / 1e6
print('Data Rate = {:1.2f} MSPS'.format(rate_msps))


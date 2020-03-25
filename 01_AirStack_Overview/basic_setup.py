# Copyright 2020 Deepwave Digital Inc.

import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
import numpy as np

rx_chan = 0             # RX1 = 0, RX2 = 1
N = 16384               # Number of complex samples per transfer
fs = 125e6              # Radio sample Rate
freq = 750e6            # LO tuning frequency in Hz
use_agc = True          # Use or don't use the AGC

#  Initialize the AIR-T receiver using SoapyAIRT
sdr = SoapySDR.Device(dict(driver="SoapyAIRT"))  # Create AIR-T instance
sdr.setSampleRate(SOAPY_SDR_RX, rx_chan, fs)     # Set sample rate
sdr.setGainMode(SOAPY_SDR_RX, rx_chan, use_agc)  # Set the gain mode
sdr.setFrequency(SOAPY_SDR_RX, rx_chan, freq)    # Tune the frequency

# Create receiver buffer
buff = np.empty(N, np.complex64)

# Turn on radio
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [rx_chan])
sdr.activateStream(rx_stream)
# Continuously read signal data from radio
while True:
    try:
        sr = sdr.readStream(rx_stream, [buff], N)
        """     Insert application code here     """
    except KeyboardInterrupt:
        """     Insert cleanup code here     """
        break
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)

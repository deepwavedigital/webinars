# Copyright 2020 Deepwave Digital Inc.
import SoapySDR, numpy
import scipy.signal as signal
import polyphase_plot

buffer_size = 2**19  # Number of complex samples per transfer
t_test = 20          # Test time in seconds
freq = 1350e6        # Tune frequency in Hz
fs = 62.5e6 / 4      # Sample rate

# Create polyphase filter
fc = 1. / max(16, 25)  # cutoff of FIR filter (rel. to Nyquist)
nc = 10 * max(16, 25)  # reasonable cutoff for our sinc-like function
win = signal.fir_filter_design.firwin(2*nc+1, fc, window=('kaiser', 0.5))

# Init buffer and polyphase filter
buff = numpy.empty(buffer_size, dtype=numpy.complex64)
s = signal.resample_poly(buff, 16, 25, window=win)

#  Initialize the AIR-T receiver using SoapyAIRT
sdr = SoapySDR.Device(dict(driver="SoapyAIRT"))     # Create AIR-T instance
sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 1, fs)     # Set sample rate
sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 1, True)     # Set the gain mode
sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 1, freq)  # Tune the frequency
rx_stream = sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32, [1])
sdr.activateStream(rx_stream)

# Run test
n_reads = int(t_test * fs / buffer_size) + 1
drop_count = 0
for _ in range(n_reads):
    sr = sdr.readStream(rx_stream, [buff], buffer_size)
    if sr.ret == -4:
        drop_count += 1
    s = signal.resample_poly(buff, 16, 25, window=win)
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)
gbps = n_reads * len(buff) * buff.itemsize * 8 / (2**30) / t_test
msg = 'Dropped Data {} Times in {:1.1f} seconds at {:1.3f} Gbps on GPU'
print(msg.format(drop_count, t_test, gbps))

polyphase_plot.psd(buff, s, fs, fs*16/25, freq, freq, title='CPU')

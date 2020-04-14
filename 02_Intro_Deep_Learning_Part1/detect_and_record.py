#!/usr/bin/env python3
# Copyright 2020 Deepwave Digital Inc.
import sys
import argparse
from SoapySDR import Device, SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW
import cusignal
import cupy as cp
from powerdetector import PowerDetector, PowerDetectorPlot, PowerDetectorWriter


def parse_command_line_arguments():
    help_formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Data Recording Tool with Detector',
                                     formatter_class=help_formatter)
    parser.add_argument('label', type=str, help='Label for recorded data')
    parser.add_argument('-n', type=int, required=False, dest='num_files',
                        default=float('inf'), help='Maximum number of files to record')
    parser.add_argument('-l', type=int, required=False, dest='seg_len', default=256,
                        help='Number of samples per file')
    parser.add_argument('-s', type=float, required=False, dest='samp_rate',
                        default=7.8128e6, help='Receiver sample rate')
    parser.add_argument('-t', type=int, required=False, dest='threshold', default=-30,
                        help='Detection threshold in dB. 0 is full scale')
    parser.add_argument('-f', type=float, required=False, dest='freq', default=315e6,
                        help='Receiver tuning frequency in Hz')
    parser.add_argument('-c', type=int, required=False, dest='channel', default=0,
                        help='Receiver channel')
    parser.add_argument('-g', type=str, required=False, dest='gain', default='agc',
                        help='Gain value')
    parser.add_argument('-b', type=int, required=False, dest='buff_len', default=32768,
                        help='Buffer size for reading from receiver')
    parser.add_argument('-d', type=int, required=False, dest='dec', default=32,
                        help='Integer decimation factor for power signal')
    parser.add_argument('-v', action='store_true', required=False, dest='visualization',
                        help='Flag show plots when signal detected')
    parser.add_argument('-p', type=str, required=False, dest='output_path',
                        default='recordings', help='Output folder for data files')
    return parser.parse_args(sys.argv[1:])


def main():
    pars = parse_command_line_arguments()

    #  Initialize the AIR-T receiver, set sample rate, gain, and frequency
    sdr = Device()
    sdr.setSampleRate(SOAPY_SDR_RX, pars.channel, pars.samp_rate)
    if pars.gain == 'agc':
        sdr.setGainMode(SOAPY_SDR_RX, pars.channel, True)  # Set AGC
    else:
        sdr.setGain(SOAPY_SDR_RX, pars.channel, float(pars.gain))  # set manual gain
    sdr.setFrequency(SOAPY_SDR_RX, pars.channel, pars.freq)

    # Create SDR shared memory buffer, detector, file writer, and plotter (if desired)
    buff = cusignal.get_shared_mem(pars.buff_len, dtype=cp.complex64)
    detr = PowerDetector(buff, pars.seg_len, pars.dec, pars.threshold)
    writer = PowerDetectorWriter(pars.output_path, pars.label, pars.num_files)
    if pars.visualization:
        plotter = PowerDetectorPlot(pars.buff_len, pars.dec, pars.samp_rate,
                                              pars.seg_len, pars.threshold)

    # Turn on radio
    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [pars.channel])
    sdr.activateStream(rx_stream)
    print('Looking for signals to record. Press ctrl-c to exit.')
    
    while True:  # Start processing Data
        try:
            sr = sdr.readStream(rx_stream, [buff], pars.buff_len)  # Read data to buffer
            if sr.ret == SOAPY_SDR_OVERFLOW:  # Data was dropped, i.e., overflow
                print('O', end='', flush=True)
            else:
                det_signal = detr.detect(buff)
                writer.tofile(det_signal)
                if pars.visualization:  # Displays the detected data if desired
                    plotter.update(cp.asnumpy(buff), detr.det_index, detr.amp_sq)
        except KeyboardInterrupt:
            sdr.deactivateStream(rx_stream)
            sdr.closeStream(rx_stream)
            break


if __name__ == '__main__':
    main()


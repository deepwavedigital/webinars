# Copyright 2020 Deepwave Digital Inc.
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import cupy as cp
import cusignal
from cusignal import filter_design


class PowerDetector:
    """ Real-time power detector class for finding signals with AIR-T
    
    This detector class is designed to do the following:
    1. Compute the instantaneous power of input signal
    2. Filter and decimate the instantaneous power to a lower data rate
    3. Reshape the down-sampled data into segments of length seg_len
    4. Perform detection on each segment of the down-sampled data
    5. Make sure at least samp_above_thresh are higher than threshold
    6. Return the segments that pass steps 4, 5
    
    Parameters
    ----------
    buff : array_like
        The input signal buffer for which to perform the power detection
    seg_len : int
        output length of segments, this is likely the length of the input layer
        to the neural network
    dec : int
        decimation factor for instantaneous power signal
    thresh_db : float
        threshold in decibels for detection in amplitude squared
    samp_above_thresh : int, optional
        Number of samples above threshold for a segment to be considered as
        having signal
    
    Examples
    --------
    The following example returns Data Rate = 151.62 MSPS on the AIR-T
    
    >>> import numpy as np
    >>> from numpy.random import randn
    >>> import cusignal
    >>> import time
    >>> from powerdetector import PowerDetector
    >>>
    >>> buff_len = 2**19
    >>> dec = 32
    >>> seg_len = 4096
    >>> threshold_db = 100
    >>> n_test = 1000
    >>>
    >>> buff = cusignal.get_shared_mem(buff_len, dtype=np.complex64)
    >>> buff[:] = randn(buff_len).astype(np.float32) + 1j*randn(buff_len).astype(np.float32)
    >>> detector = PowerDetector(buff, buff_len, dec, threshold_db)
    >>> t0 = time.monotonic()  # Start timer
    >>> for _ in range(n_test):  # Run step n_buffer times
    >>>     output_segments = detector.detect(buff)
    >>> rate_msps = buff_len * n_test / (time.monotonic() - t0) / 1e6
    >>> print('Data Rate = {:1.2f} MSPS'.format(rate_msps))
    """
    
    def __init__(self, buff, seg_len, dec, thresh_db, samp_above_thresh=4):
        assert samp_above_thresh <= (seg_len / dec), 'decimated seg_len shorter than ' \
                                                     'samp_above_thresh'
        self._seg_len = seg_len
        self._dec = dec
        self._win = self._create_fir_filter_window()
        self._seg_len_dec = int(self._seg_len / self._dec)
        self._thresh = 10 ** (thresh_db / 10)  # Convert thresh to linear units
        self._samp_above_thresh = samp_above_thresh
        self._x_power_dec = cp.zeros(int(len(buff) / dec), dtype=cp.float32)
        self._seg_det_index = cp.zeros(int(len(buff) / seg_len), dtype=bool)
        self.detect(buff)  # Run detector one time to compile the CUDA kernels
    
    def _create_fir_filter_window(self):
        """ Creates FIR filter coefficients
        
        Returns
        -------
        win : array_like
            2D CuPy array with FIR filter coefficients
        """
        ntaps = 2 * self._dec + 1
        cut = 1 / self._dec
        filt_coef = filter_design.fir_filter_design.firwin(ntaps, cut,
                                                           window=('kaiser', 0.5))
        win = cp.asarray(filt_coef, dtype=cp.float32)
        return win
    
    def detect(self, x):
        """ Calculates instantaneous power of signal and performs detection
        
        Parameters
        ----------
        x : array_like
            The input signal for which to perform the power detection
        
        Returns
        -------
        y : array_like
            2D CuPy array of shape (m, seg_len) where m is the number of
            detections found
        """
        
        # Compute the instantaneous power of the x signal (full data rate)
        x_power = cp.power(cp.abs(x), 2)
        
        # Filter and decimate the power to a lower data rate
        self._x_power_dec[:] = cusignal.decimate(x_power, self._dec, n=self._win,
                                                 zero_phase=True)
        
        # Reshape the down-sampled data into a matrix where rows are segments
        x_power_dec_mat = self._x_power_dec.reshape(-1, self._seg_len_dec)
        
        # Perform detection on each row (segment) of the down-sampled data
        x_det_dec_mat = x_power_dec_mat > self._thresh
        
        # Make sure at least samp_above_thresh are higher than the threshold
        self._seg_det_index[:] = cp.sum(x_det_dec_mat, axis=1) > self._samp_above_thresh
        
        # Reshape the input signal to be of shape (m, seg_len) and remove
        # segments without a detection
        y = x.reshape(-1, self._seg_len)[cp.asnumpy(self._seg_det_index)]
        return y
    
    @property
    def amp_sq(self):
        """ Amplitude Square of the Signal
        
        Returns
        -------
        ndarray : instantaneous power of the x signal at decimated rate
        """
        return cp.asnumpy(self._x_power_dec)
    
    @property
    def det_index(self):
        """ Index of Segments with Detections
        
        Returns
        -------
        ndarray : segment detection index
        """
        return cp.asnumpy(self._seg_det_index)


class PowerDetectorPlot:
    """ Plotting class to visualize the PowerDetector class
    
    Parameters
    ----------
    buff_len : int
        length of the full bandwidth data buffer (top plot)
    dec : int
        decimation factor for instantaneous power signal
    samp_rate: float
        sample rate of full bandwidth data buffer (top plot)
    seg_len : int
        output length of segments, this is likely the length of the input layer
        to the neural network
    thresh_db : float
        threshold in decibels for detection in amplitude squared
    top_fill_on : bool
        Fill in detection segments on full bandwidth data plot (top)
    bot_fill_on : bool
        Fill in detection segments on detection plot (bottom)
    top_ylim : array_like
        y axis limits for full bandwidth data plot (top)
    bot_ylim : array_like
        y axis liimits for detection plot (bottom)
        
    Examples
    --------
    >>> plotter = PowerDetectorPlot(n0, fs0, n1, fs1, thresh, input_len)
    >>> while True:
    >>>     sig = create_signal()  # Your function
    >>>     det_idx = perform_detection(sig)  # Your function
    >>>     plotter.update(det_idx)  # det_idx indices where power > thresh
    
    """
    
    def __init__(self, buff_len, dec, samp_rate, seg_len, thresh_db, top_fill_on=False,
                 bot_fill_on=True, top_ylim=(-1, 1), bot_ylim=(-50, 0)):
        # Create figure
        plt.style.use('dark_background')
        plt.ion()
        self._top_fill_on = top_fill_on  # Show detection regions on top plot
        self._bot_fill_on = bot_fill_on  # Show detection regions on bottom plot
        self._fig, ax = plt.subplots(2, 1, figsize=(8.5, 11), sharex='col', dpi=75)
        self._fig.tight_layout()
        
        # Setup I/Q plot (Top)
        x_top = np.arange(0, buff_len) / samp_rate / 1e-3  # msec
        if self._top_fill_on:
            top_fill_xyc = self._get_fill_x_y_color(top_ylim, buff_len, seg_len, x_top)
            self._top_fill = ax[0].fill(*top_fill_xyc)
        self._top_imag, = ax[0].plot(x_top, np.zeros_like(x_top), 'orange',
                                     label='complex')
        self._top_real, = ax[0].plot(x_top, np.zeros_like(x_top), '#70bf4d',
                                     label='real')
        self._title = ax[0].set_title('')
        ax[0].legend(loc=1)
        ax[0].set_xlim(x_top[0], x_top[-1])
        ax[0].set_ylim(top_ylim)
        ax[0].set_ylabel('Complex Signal')
        
        # Setup detector plot (bottom)
        nsamples_dec = int(buff_len / dec)
        fs_dec = samp_rate / dec
        x_bot = np.arange(0, nsamples_dec) / fs_dec / 1e-3
        if self._bot_fill_on:
            bot_fill_xyc = self._get_fill_x_y_color(bot_ylim, buff_len, seg_len, x_top)
            self._bot_fill = ax[1].fill(*bot_fill_xyc)
        self._bot_pow, = ax[1].plot(x_bot, np.zeros_like(x_bot), color='fuchsia',
                                    label='Power')
        ax[1].plot([x_bot[0], x_bot[-1]], [thresh_db, thresh_db], '--', linewidth=2,
                   color='w', label='Threshold')
        ax[1].legend(loc=1)
        ax[1].set_ylim(bot_ylim)
        ax[1].set_ylabel('Power Detector')
    
    @staticmethod
    def _get_fill_x_y_color(ylims, buff_len, seg_len, x_vals):
        """ Creates a tuple of x_top, y_top, color_top, x_bot, y_bot, color_bot
        
        Creates a tuple of x_top, y_top, color_top, x_bot, y_bot, color_bot. Each
        x, y, color is the data to fill that segment. We can toggle the segments
        on/off with the alpha parameter if a signal was detected.
        
        Parameters
        ----------
        ylims : array_like
            y axis limits for full bandwidth data plot (top)
        buff_len : int
            length of the full bandwidth data buffer (top plot)
        seg_len : int
            output length of segments
        x_vals : array_like
            array of x values in plot
         
        Returns
        -------
        xyc_vals : tuple
            tuple of x,y bounds and color values for each segment
        """
        
        nst = np.arange(0, buff_len, seg_len)
        xyc_vals = []
        color_list = ['teal', '#0d4175']
        for i, xst in enumerate(nst):
            xen = xst + seg_len - 1
            xyc_vals.append((x_vals[xst], x_vals[xen], x_vals[xen], x_vals[xst]))
            xyc_vals.append((ylims[0], ylims[0], ylims[1], ylims[1]))
            xyc_vals.append(color_list[i % 2])
        return xyc_vals
    
    def update(self, sig_cplx, det_idx, amp_sq):
        """ Updates the plot data for the PowerDetector Class
        
        Parameters
        ----------
        sig_cplx : ndarray
            complex valued signal array
        det_idx : ndarray
            boolean array of detection segments
        amp_sq : ndarray
            real valued array of instantaneous power signal

        """
        
        if det_idx.any():  # Only update plot if there was a detection
            # Update title
            self._title.set_text('{} Files Saved to Disk'.format(np.sum(det_idx)))
            
            # Update detection fill segments by toggling the segment fills on/off based
            # on if signal was detected
            if self._top_fill_on:
                for ax, on_off in zip(self._top_fill, det_idx):
                    ax.set_alpha(float(on_off))
            if self._bot_fill_on:
                for ax, on_off in zip(self._bot_fill, det_idx):
                    ax.set_alpha(float(on_off))
            
            # Update complex signal plot by changing the y-data
            self._top_real.set_ydata(sig_cplx.real)
            self._top_imag.set_ydata(sig_cplx.imag)
            
            # Update setector plot by changing the y-data
            self._bot_pow.set_ydata(10 * np.log10(np.abs(amp_sq)))
            
            # Draw the updated plot and display it
            plt.draw()
            plt.pause(.001)


class PowerDetectorWriter:
    """ Writes the detected signals to disk for the PowerDetector
    
    Parameters
    ----------
    output_path : str
        location to save data files
    label : str
        folder in output_path to save files
    num_files : int
        maximum number of files to record before terminating
    """
    
    def __init__(self, output_path, label='file', num_files=float('inf')):
        self._num_files = num_files
        if self._num_files > 0:
            self._output_path = os.path.join(output_path, label)
            self._label = label
            self._ctr = 0
            os.makedirs(self._output_path, exist_ok=True)
    
    def tofile(self, signal_matrix):
        """ Write to disk
        
        Parameters
        ----------
        signal_matrix : array_like
            matrix of signal data to write to disk. Rows written to individual files
        """
        
        for i, sig in enumerate(signal_matrix):
            filename = '{}_{:010.0f}.bin'.format(self._label, self._ctr)
            sig.tofile(os.path.join(self._output_path, filename))
            self._ctr += 1
            if self._ctr >= self._num_files:
                print('File Write counter = {}. Exiting.'.format(self._ctr))
                sys.exit(0)
            elif i == len(signal_matrix)-1:  # Print if last write
                print('File Write counter = {}'.format(self._ctr))

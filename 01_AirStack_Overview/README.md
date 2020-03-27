# Deep Learning and Signal Processing Webinar

## Author
<p align="center">
<img src="http://www.deepwavedigital.net/logos/deepwave-logo-2-white.png" Width="50%" />
</p>

This software is written by **Deepwave Digital, Inc.** [www.deepwavedigital.com]().

## Inquiries
  - General company contact: [https://deepwavedigital.com/inquiry](https://deepwavedigital.com/inquiry)
&nbsp;


## Webinar Description
![](https://deepwavedigital.com/media/2020/cusignal_on_airt.gif)

&nbsp;

**Date:** Streamed live March 25, 2020

**Slides**: [Deepwave_Webinar_20200325_Public_Release.pdf](https://deepwavedigital.com/presentations/2020/Deepwave_Webinar_20200325_Public_Release.pdf)

**Video:** [https://youtu.be/S17vUaTDHts](https://youtu.be/S17vUaTDHts)

**Topics**
1. Overview of Deepwave Digital
2. Using the AIR-T for Machine Learning in RF systems
3. Demonstration of Signal Processing using the GPU on the AIR-T
4. Demonstration of using cuSignal
5. Application of Deep Learning

<br>

## Source Code Used in Webinar

### AirStack Radio Drivers Demonstration

The demonstration walked the audience through the basic source code framework for the AIR-T Radio.

* **Source Code**:
  * `basic_setup.py`

### NVIDIA RAPIDS and CuSignal with the AIR-T Demonstration

* Install Anaconda on the AIR-T: [http://docs.deepwavedigital.com/Tutorials/6_conda.html](http://docs.deepwavedigital.com/Tutorials/6_conda.html)
* Install cuSignal on the AIR-T: [http://docs.deepwavedigital.com/Tutorials/7_cuSignal.html](http://docs.deepwavedigital.com/Tutorials/7_cuSignal.html)
* **Source Code**:
  * `polyphase_cpu.py` - *scipy.signal* polyphase resampler
  * `polyphase_gpu.py- cuSignal polyphase resampler`
  * `polyphase_plot.py - Plotting utility`

![](https://deepwavedigital.com/media/2020/cpu_vs_gpu_diff.png)

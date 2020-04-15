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

**Date:** Streamed live April 14, 2020

**Slides**: [02_Intro_Deep_Learning_Part1.pdf](https://deepwavedigital.com/media/2020/02_Intro_Deep_Learning_Part1.pdf)

**Video:** Coming Soon

**Topics**
1. Overview of Deepwave Digital
2. Using the AIR-T for Machine Learning in RF systems
3. Signal Processing with cuSignal
4. Training Data Acquisition

<br>

## Source Code Used in Webinar

### Signal Processing with cuSignal Demonstration

The demonstration walked the audience through the more advanced usage of the cuSignal
API on the AIR-T Radio.

* Install Anaconda on the AIR-T: [http://docs.deepwavedigital.com/Tutorials/6_conda.html](http://docs.deepwavedigital.com/Tutorials/6_conda.html)
* Install cuSignal on the AIR-T: [http://docs.deepwavedigital.com/Tutorials/7_cuSignal.html](http://docs.deepwavedigital.com/Tutorials/7_cuSignal.html)

* **Source Code**:
  * `power_bench_cpu_vs_gpu.py`
  * `power_bench_compile.py`

### Training Data Acquisition Demonstration


* **Source Code**:
  * `powerdetector.py`
  * `powerdetector_bench.py`
  * `detect_and_record`

![](https://deepwavedigital.com/media/2020/detect_and_record.png)


## Basic setup and Installation

## Requirements:

1. Version of CuSignal with PR #60 integrated. If that hasn't happend yet clone this repo:
```
$ cd ~/
$ git clone https://github.com/deepwavedigital/cusignal.git
$ git checkout remotes/origin/decimate-resuse-fir-coef
```

2. Create the conda environment
```
$ cd cusignal
$ conda env create -f conda/environments/cusignal_base.yml
```

3. Activate environment and install cuSignal
```
$ conda activate cusignal
$ cd python
$ python setup.py install
```

4. Check out webinar code
```
$ cd ~/
$ git clone https://github.com/deepwavedigital/webinar-dev.git
$ cd webinar-dev/02_Intro_Deep_Learning_Part1/
$ python powerdetector_bench.py
```

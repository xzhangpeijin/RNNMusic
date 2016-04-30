# RNNMusic

Synthetic music generation using LSTMs

Peijin Zhang and Taehoon Lee, 15780 Term Project

## Overview

We aim to train LSTMs over a corpus of audio files and use the trained models to synthsize new music. In contrast to previous approaches, we use raw audio feature representations instead of notes. This gives our models additional flexibility and generalizability to all music/audio types.

### Audio transformation

We read raw audio wave files and perform STFT transforms to give us amplitudes over the frequency domain. Amplitude and phase information is stored per frequency measurement in the feature representation.

### Feature Representation

Each encoding is a csv file. The first column is in the form:

    <Sampling Rate>,<First bin start>,<Next bin start> ... <First bin start>,<Next bin start>, ...
	time, amplitude_1, amplitude2, ... , angle_1, angle_2, ...
	...
---
title: 'shroom: A Python framework for Ambisonics room acoustics simulation and binaural rendering'
tags:
  - Python
  - spatial audio
  - Ambisonics
  - spherical harmonics
  - room acoustics
  - binaural rendering
  - HRTF
authors:
  - name: Yhonatan Gayer
    orcid: 0009-0009-1156-9087
    affiliation: 1
affiliations:
  - name: School of Electrical and Computer Engineering, Ben-Gurion University of the Negev, Israel
    index: 1
date: 11 July 2026
bibliography: paper.bib
---

# Summary

`shroom` (Spherical Harmonics ROOM) is an open-source Python library for simulating room
acoustics in the spherical-harmonics (SH) / Ambisonics domain and rendering the result to
binaural audio. It computes image-source reflections with the image-source method
[@allen1979JASA] and projects every source onto an SH basis
[@rafaely2015SphArray; @zotter2019Ambisonics] in a single batched step, producing an Ambisonic
Room Impulse Response (ARIR). All downstream processing — binaural decoding with
head-related transfer functions (HRTFs), spherical microphone-array simulation, Ambisonics
Signal Matching (ASM) [@gayer2024ICASSPW] and Binaural Signal Matching (BSM) [@madmoni2024arXiv]
encoding, and dynamic listener head rotation via Wigner-D matrices [@magariyachi2020JASA] — operates on
this same ARIR through a small set of composable processors.

The core abstraction is a `SpatialSignal` object carrying data of shape
`(n_channels, n_spatial, n_samples)` together with two lazy, in-place domain flags
(time/frequency and space/SH). Processors implement a uniform
`process(SpatialSignal) -> SpatialSignal` interface and can be chained; a `ProcessorChain`
collapses a sequence of SH-domain filters into a single kernel, avoiding redundant FFTs.
A bundled HRTF dataset lets users run the full pipeline immediately after
`pip install pyshroom`.

# Statement of need

Room-acoustics simulation in Python is well served by `pyroomacoustics`
[@scheibler2018ICASSP], which provides an efficient image-source engine, ray
tracing, and array-processing tools. `shroom` builds directly on its image-source geometry.
However, `pyroomacoustics` never enters the SH domain: its binaural path relies on
nearest-neighbour HRTF selection, and every change of listener orientation forces
re-accumulation of all $O(R^3)$ image sources at reflection order $R$. This makes it
unsuitable for the SH-domain workflows that are now standard in spatial-audio research —
Magnitude Least Squares (MagLS) HRTF pre-processing [@schorkhuber2018DAGA; @lubeck2020JAES],
Wigner-D head rotation, spherical-array modelling [@rafaely2005TASP], and composable
SH-domain encoders.

`shroom` fills this gap. By projecting all image sources onto the SH basis once, binaural
decoding reduces to a single matrix–filter product that is independent of source count, so
the decode cost is paid once and amortises over multiple sources and over head orientations.
On top of the shared ARIR it provides capabilities absent from `pyroomacoustics`: MagLS and
array-aware MagLS rendering [@gayer2026TASLP], real-time Wigner-D head rotation, rigid/open spherical
microphone-array simulation with configurable radial models, and ASM/BSM encoders for
arbitrary arrays. This makes `shroom` a practical research tool for developing and evaluating
Ambisonics capture, encoding, and binaural-reproduction algorithms within a single
consistent Python API.

The accompanying preprint [@gayer2026shroom] reports a detailed evaluation. In brief, MagLS
rendering reaches perceptual transparency at low SH orders — approximately 2 dB log-spectral
distance to a high-order reference at order 5, within the reported 1–2 dB just-noticeable
difference [@benhur2017spectral; @engel2022AcuActa] — and cached Wigner-D head rotation costs well
under 1 ms per frame at order 3. The library is intended for spatial-audio and acoustics
researchers, and is suitable for teaching Ambisonics and SH signal processing.

# Acknowledgements

The bundled HRTF data is the Neumann KU 100 spherical far-field HRIR compilation
[@hrir2020KU100]. Room geometry and image-source computation build on `pyroomacoustics`
[@scheibler2018ICASSP], and the MagLS implementation follows the formulation of
Schörkhuber et al. [@schorkhuber2018DAGA].

# References

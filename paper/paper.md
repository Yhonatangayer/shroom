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

Research in Ambisonics and binaural spatial audio increasingly operates in the
spherical-harmonics domain: sound fields are represented by SH coefficients, and rendering,
head rotation, and array encoding are formulated as operations on those coefficients. Yet
the open-source Python tools available to this community do not natively support the SH
domain. Researchers therefore assemble ad-hoc pipelines that stitch together a room
simulator, a separate HRTF processing step, and hand-written SH and array code — an
error-prone process that makes results hard to reproduce and algorithms hard to compare on a
common footing.

`shroom` addresses this need by providing a single, consistent SH-domain pipeline that spans
room simulation, binaural rendering, and microphone-array capture. Its target users are
spatial-audio and acoustics researchers who develop and evaluate Ambisonics capture,
encoding, and binaural-reproduction algorithms, as well as instructors teaching Ambisonics
and SH signal processing. By projecting all image sources onto the SH basis once, `shroom`
turns binaural decoding into a single matrix–filter product that is independent of source
count, so the decode cost is paid once and amortises over multiple sources and over head
orientations. On top of this shared representation it supplies the building blocks that
SH-domain research requires — MagLS and array-aware MagLS rendering [@gayer2026TASLP],
real-time Wigner-D head rotation, rigid and open spherical microphone-array simulation with
configurable radial models, and ASM/BSM encoders for arbitrary arrays — within one Python
API.

# State of the field

Room-acoustics simulation in Python is well served by `pyroomacoustics`
[@scheibler2018ICASSP], which provides an efficient image-source engine, ray tracing, and
array-processing tools; `shroom` builds directly on its image-source geometry. However,
`pyroomacoustics` never enters the SH domain: its binaural path relies on nearest-neighbour
HRTF selection, and every change of listener orientation forces re-accumulation of all
$O(R^3)$ image sources at reflection order $R$. It therefore does not support the SH-domain
workflows that are now standard in spatial-audio research — Magnitude Least Squares (MagLS)
HRTF pre-processing [@schorkhuber2018DAGA; @lubeck2020JAES], Wigner-D head rotation,
spherical-array modelling [@rafaely2005TASP], and composable SH-domain encoders.

Rather than reimplement room acoustics, `shroom` *contributes* the missing SH-domain layer
on top of an established simulator: it reuses `pyroomacoustics` for geometry and image-source
computation and adds the SH projection, HRTF decoding, head rotation, array simulation, and
encoding stages as first-class, composable operations. To our knowledge no existing
open-source Python package couples image-source room simulation with a batched SH
representation and this range of array-aware rendering and encoding tools in one interface.
This "contribute, don't rebuild" choice keeps `shroom` focused on its distinct scholarly
contribution — an efficient, reproducible SH-domain pipeline for spatial-audio research.

# Software design

The core abstraction is the `SpatialSignal` object, which carries data of shape
`(n_channels, n_spatial, n_samples)` together with two lazy, in-place domain flags: a
time/frequency flag and a space/SH flag. Rather than eagerly transforming data, `shroom`
tracks the current domain and defers each FFT or SH transform until an operation actually
requires it. This design choice trades a small amount of bookkeeping for the elimination of
redundant transforms, which dominate cost in multi-stage SH pipelines.

Processing is expressed through a uniform `process(SpatialSignal) -> SpatialSignal` interface,
so every stage — SH projection, HRTF decoding, array decoding, encoding, head rotation — is a
composable processor. A `ProcessorChain` exploits this uniformity by collapsing a sequence of
SH-domain filters into a single equivalent kernel, so a chain of filters is applied as one
matrix–filter product instead of repeated forward/inverse transforms. The main trade-off is
generality versus specialisation: the shared `SpatialSignal` contract constrains what a
processor may assume about its input, but in return any processor composes with any other and
the batched SH projection makes the decode cost independent of source count — the property
that makes the whole pipeline efficient for research use. A bundled HRTF dataset lets users
run the full pipeline immediately after `pip install pyshroom`, lowering the barrier to
reproducing and extending experiments.

# Research impact statement

`shroom` implements and reproduces the methods evaluated in an accompanying preprint
[@gayer2026shroom] and in related work on array-aware Ambisonics and HRTF encoding
[@gayer2026TASLP; @gayer2024ICASSPW]. That evaluation shows MagLS rendering reaching
perceptual transparency at low SH orders — approximately 2 dB log-spectral distance to a
high-order reference at order 5, within the reported 1–2 dB just-noticeable difference
[@benhur2017spectral; @engel2022AcuActa] — while cached Wigner-D head rotation costs well
under 1 ms per frame at order 3. The repository ships runnable examples and convergence
benchmarks for ASM, BSM, and array-aware MagLS, so these results can be reproduced directly.
The library is packaged on PyPI, tested, and documented with contribution guidelines, making
it ready for reuse and community contribution by spatial-audio and acoustics researchers and
as teaching material for Ambisonics and SH signal processing.

# AI usage disclosure

Generative AI tools were used to assist with software documentation, code refactoring, and
the preparation of this paper (including editing for clarity and structure). All
AI-suggested content was reviewed by the author, and correctness was verified through the
project's automated test suite and continuous-integration pipeline together with manual
inspection; the algorithms, design decisions, and reported results are the author's own. No
generative AI was used to produce the experimental results.

# Acknowledgements

The bundled HRTF data is the Neumann KU 100 spherical far-field HRIR compilation
[@hrir2020KU100]. Room geometry and image-source computation build on `pyroomacoustics`
[@scheibler2018ICASSP], and the MagLS implementation follows the formulation of
Schörkhuber et al. [@schorkhuber2018DAGA].

# References

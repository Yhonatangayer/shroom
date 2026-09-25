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
  - ATF
authors:
  - name: Yhonatan Gayer
    orcid: 0009-0009-1156-9087
    affiliation: 1
  - name: Boaz Rafaely
    affiliation: 1
affiliations:
  - name: School of Electrical and Computer Engineering, Ben-Gurion University of the Negev, Israel
    index: 1
date: 11 July 2026
bibliography: paper.bib
---

# Summary

Spatial audio research for virtual and augmented reality, teleconferencing and hearing devices often represents sound fields in the Spherical Harmonics (SH) domain, known as Ambisonics. We present SHroom (Spherical Harmonics ROOM), an open-source Python library that covers the SH-domain research workflow in a single package: (i) simulating a room and generating its Ambisonic Room Impulse Response (ARIR) with the image-source method; (ii) convolving dry audio with the ARIR to obtain Ambisonics signals; (iii) filtering these signals in the SH domain with Head-Related Transfer Functions (HRTFs) or microphone-array Acoustic Transfer Functions (ATFs) to produce ear or microphone signals; and (iv) further SH-domain processing, namely Wigner-D head rotation, simulation of spherical microphone arrays, HRTF preprocessing with MagLS and array-aware MagLS, and Ambisonics encoding from arbitrary array geometries with Ambisonics signal matching (ASM) and Binaural signal matching (BSM).

Most existing tools cover either stages (i)–(ii) or stages (iii)–(iv), and those that span several stages provide them as separate modules rather than one chain, so researchers bridge them with ad-hoc code that is hard to reproduce and compare. In SHroom, all four stages operate on one shared signal type through one processing interface, so a simulated room flows through the complete chain without format conversions. Built on the image-source engine of `pyroomacoustics`, SHroom reproduces the ARIR of its spherical-harmonic receivers while computing it about 3x faster for SH orders 4 to 12. SHroom is available at [https://github.com/Yhonatangayer/shroom](https://github.com/Yhonatangayer/shroom) and installable via `pip install pyshroom`.

# Statement of need

The development and evaluation of spatial audio algorithms based on Ambisonics rely on the ability to simulate acoustic environments, process Ambisonics and other types of signals, and render binaural or loudspeaker signals. A typical research workflow encompasses various and diverse computational components: (i) the generation of Ambisonic Room Impulse Responses (ARIRs), (ii) the generation of Ambisonics signals by convolving dry audio signals with the ARIRs, (iii) Spherical Harmonics (SH) domain filtering, such as convolution with Head-Related Transfer Functions (HRTFs) or microphone array Acoustic Transfer functions (ATFs) to produce signals at the ears or at the microphones, (iv) additional SH-domain processing including operations such as Wigner-D based listener head rotations, modeling of spherical microphone array prototypes, processing of SH-domain HRTFs, and encoding of Ambisonics from microphones arrays. 

Implementing all these computational components within a single computation tool would streamline the research workflow. However, existing software can only provide implementations for various parts of such workflow.

# State of the field

For stages (i) and (ii), several existing room-acoustics simulators can generate ARIRs and convolve them with dry audio signals to produce Ambisonic signals. As summarized in Table 1, `pyroomacoustics` (PRA) [@pyroomacoustics-Scheibler2018], MASP [@perezlopez2020AES], shoebox-roomsim [@politis2016roomsim], MCRoomSim [@wabnitz2010ISRA], and SAF [@saf2024framework] generate ARIRs using the image-source method (ISM), whereas GSound-SIR [@zang2025GSoundSIR] employs ray tracing (RT). These tools therefore cover room simulation and Ambisonics signal generation, but differ in the downstream processing they offer, as detailed in (iii) and (iv) above.

For stage (iii), SH-domain filtering, such as convolution with HRTFs or ATFs support is fragmented (see column 'SH domain ATF processing' in Table 1). Tools such as MASP [@perezlopez2020AES], SAF [@saf2024framework], `spaudiopy` [@hold2025spaudiopy], `sound-field-analysis-py`, and `pyfar`/`spharpy` provide dedicated SH-domain ATF processing. In contrast, room simulators like `shoebox-roomsim` [@politis2016roomsim] and `MCRoomSim` [@wabnitz2010ISRA] do not provide post-simulation SH-domain ATF filtering.

Stage (iv) encompasses additional processing such as HRTF pre-processing (e.g. MagLS equalization), the application of SH rotation matrices for listener head tracking, modeling of open and rigid spherical microphone array prototypes, and Ambisonics array encoding. Specialized toolboxes such as `spaudiopy` [@hold2025spaudiopy] provide these capabilities, but omit room-acoustic engines altogether. Other packages cover only a subset of these features; for example, `pyfar`/`spharpy` lacks MagLS optimization, MASP [@perezlopez2020AES] and `sound-field-analysis-py` do not include native rotation matrices, and room simulators (`pyroomacoustics`, `GSound-SIR`, `shoebox-roomsim`, `MCRoomSim`) bypass stage (iv) entirely. While SAF [@saf2024framework] supports a comprehensive feature set, it does so through standalone C/C++ modules rather than a unified, scriptable workflow. In conclusion, no toolbox is currently available which offers this wide range of computations, which may be important in spatial audio processing research. 

## Contributions

To address the fragmented availability of tools within this spatial audio workflow, this paper introduces SHroom, an open-source Python library designed to unify room acoustics and SH processing. The main contributions of this work are:

* **An end-to-end SH-domain pipeline:** SHroom bridges the gap between room-acoustic simulation (stages i–ii) and downstream spatial processing (stages iii–iv). It is the first Python framework that allows users to generate ARIRs and seamlessly use them in advanced SH-domain processing without relying on external wrappers.
* **Comprehensive stage (iv) capabilities:** SHroom natively integrates SH processing. Each module implements an established method, all of which operate on the same ARIR and compose seamlessly with one another. Capabilities include Wigner-D rotation matrices, simulation of prototype and arbitrary microphone arrays, robust spatial encoding from arbitrary array geometries via the Ambisonics Signal Matching (ASM) [@ASM; @Parametric-ASM-like-paper] and Binaural Signal Matching (BSM) [@BSM_journal_paper; @Shai-paper], and HRTF preprocessing via MagLS [@HRTF_MagLS; @kassakian2006convex; @Ambisonics_MagLS] and array-aware MagLS HRTF (AA-MagLS) [@gayer2026TASLP]. 

### Table 1: Capability comparison of open-source spatial-audio software

| Software | Language | Room sim. & ARIR | SH domain ATF processing | MagLS | SH rotation | SH array simulation | Array encoding |
|----------------------|----------|------------|-----------|--------|---------|----------|---------|
| pyroomacoustics | Python | $\checkmark$ (ISM) | - | - | - | - | - |
| GSound-SIR | Python | $\checkmark$ (RT) | - | - | - | - | - |
| MASP | Python | $\checkmark$ (ISM) | $\checkmark$ | - | - | $\checkmark$ | $\checkmark$ |
| shoebox-roomsim | MATLAB | $\checkmark$ (ISM) | - | - | - | - | - |
| MCRoomSim | MATLAB | $\checkmark$ (ISM) | - | - | - | - | - |
| SAF | C / C++ | $\checkmark$ (ISM) | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| spaudiopy | Python | - | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| sound-field-analysis-py | Python | - | $\checkmark$ | - | - | $\checkmark$ | $\checkmark$ |
| pyfar / spharpy | Python | - | $\checkmark$ | - | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **SHroom** | **Python** | **$\checkmark$ (ISM)** | **$\checkmark$** | **$\checkmark$** | **$\checkmark$** | **$\checkmark$** | **$\checkmark$** |

*Note: $\checkmark$ = provided as a first-class feature; - = not provided. "Language" refers to the user-facing interface language of the software. "Room sim. & ARIR" means the simulated room field is exposed to the user as SH coefficients. "SH array simulation" means modal simulation of rigid or open spherical arrays with radial filters, not merely the placement of pressure microphones in a room. "Array encoding" means encoding those signals from arbitrary array geometries. SAF provides the individual capabilities as independent C modules rather than as one composable pipeline.*

# Software design

SHroom is organised around one data type and one interface. `SpatialSignal` holds a multichannel signal together with its sampling rate, spatial grid and two domain flags (time or frequency; space or SH), and records every transform applied to it. Domain changes (`toFreq`, `toSH`, `toSpace`) and rotations (Wigner-D in the SH domain, or grid rotation in the space domain) are methods on this type, so every module accepts and returns the same object and checks the domain it needs rather than assuming it.

Processing steps (`BinauralDecoder`, `ArrayDecoder`, `ASMEncoder`, `BSMEncoder`) implement a single `process` method from `SpatialSignal` to `SpatialSignal`. They can therefore be applied in any order that is physically meaningful, and a `ProcessorChain` collapses a sequence of them into one equivalent filter by passing unit impulses through the chain once, so a long signal is convolved only a single time.

The room model deliberately reuses the `pyroomacoustics` image-source engine for geometry, wall materials and reflection bookkeeping instead of re-implementing it, and replaces only the receiver stage: all image sources are projected onto the SH basis in one vectorised evaluation and placed with the same fractional-delay filters, rather than simulating one directional microphone per SH channel. Frequency-dependent absorption is handled per octave band, as in `pyroomacoustics`. The trade-offs are explicit: SHroom supports shoebox rooms and the image-source method only, and it rejects `ray_tracing=True` instead of silently ignoring it; randomised image-source positions are enabled by default to reduce the metallic sound of the sparse late field, and `Room.ism_coverage()` warns when the reflection order is too low for the room's reverberation time.

# Research impact statement

SHroom provides open, tested Python implementations of ASM [@ASM], BSM [@BSM_journal_paper], MagLS [@HRTF_MagLS] and AA-MagLS [@gayer2026TASLP]. Its correctness is checked against independent references: driven from the same image-source engine, its ARIR reproduces the one obtained from `pyroomacoustics`' spherical-harmonic receivers up to the float32 precision of the `pyroomacoustics` output, while SHroom computes it in float64 and about 3x faster for $4 \le N \le 12$, and its BSM encoder is validated against a MATLAB reference. The repository ships benchmark scripts for this comparison and for the convergence of the ASM, spectrally equalized ASM, BSM and AA-MagLS encoders with SH order, and six runnable examples covering each binaural rendering path (HRTF, MagLS, ASM, ASM with AA-MagLS, BSM, and head rotation).

SHroom is released on PyPI (`pip install pyshroom`), where it has been downloaded about 1,000 times since its first release in March 2026 (pypistats, excluding mirrors). It is tested by a 212-test suite in continuous integration across Python 3.9–3.13, and ships contribution guidelines and citation metadata.

# AI usage disclosure

Generative AI tools were used to assist with software documentation, code refactoring, and
the preparation of this paper (including editing for clarity and structure). All
AI-suggested content was reviewed by the authors, and correctness was verified through the
project's automated test suite and continuous-integration pipeline together with manual
inspection; the algorithms, design decisions, and reported results are the authors' own. No
generative AI was used to produce the experimental results.

# Acknowledgements

The bundled HRTF data is the Neumann KU 100 spherical far-field HRIR compilation
[@HRTF_data_set]. Room geometry and image-source computation build on `pyroomacoustics`
[@pyroomacoustics-Scheibler2018], and the MagLS implementation follows the formulation of
Schörkhuber et al. [@HRTF_MagLS].

# References

# Kidney CT Contrast Timing for Function Screening

This repository contains code developed for an MSc thesis focused on estimating **contrast timing** in routine abdominal CT scans to support **kidney function screening**.

## Overview

The project uses machine learning to estimate contrast timing from CT images:
1. **Global phase classification**: Predict the overall contrast phase (e.g., arterial / portal venous / late)
2. **Within-phase timing estimation**: Estimate a continuous early→late timing index within each phase to support phase-aware correction of renal enhancement measurements

## Motivation
Abdominal CT examinations often use **iodinated contrast**, which is primarily cleared by the kidneys. In principle, the way contrast appears in the renal vessels and tissue contains information about renal function and perfusion. However, even when scans are acquired using standardized protocols (e.g., arterial / portal venous / late), **the same nominal phase does not guarantee the same physiological timing across patients**. Inter-patient variability in contrast dynamics (cardiac output, blood volume, renal blood flow, injection differences, etc.) leads to timing mismatch, which complicates quantitative measurements.

A key challenge is the **wavefront effect**: when contrast is still propagating and mixing, measurements taken from different anatomical regions can be biased simply because the bolus has not fully equilibrated. If we can estimate where a scan lies along the contrast progression (globally and within the kidneys), it may be possible to standardize enhancement measurements and make downstream functional metrics more consistent.

## Project goal
The goal of the thesis is to develop machine-learning methods to estimate contrast timing from CT images, with two main components:

1. **Global phase estimation (supervised)**  
   Predict the overall contrast phase of a series (e.g., arterial / portal venous / late).

2. **Within-phase timing estimation (weakly supervised / unsupervised)**  
   Estimate a continuous **early→late timing index** within each nominal phase, with a focus on kidney-specific contrast progression. This is intended to support *phase-aware correction* of renal enhancement measurements and reduce wavefront-driven variability.

## Clinical relevance
If contrast timing can be estimated reliably from scans that are already being acquired for other clinical reasons, it could enable **opportunistic screening** of declining renal function without additional tests, radiation, or dedicated clearance studies—while also supporting research into more quantitative renal clearance measures from CT.


## Setup Instructions

### Option 1: Using Conda (Recommended for Full Reproducibility)

For exact reproducibility with all binary dependencies and build hashes:

```bash
conda env create -f environment.yml
conda activate kidney
```

To update an existing environment:
```bash
conda env update -f environment.yml --prune
```

### Option 2: Using pip (More Portable)

For a lighter setup with flexible version ranges:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies Summary

**Core Libraries:**
- `numpy`, `scipy`, `pandas`: Numerical and data processing
- `SimpleITK`: DICOM loading and medical image processing  
- `nibabel`: NIfTI file I/O
- `matplotlib`, `PIL`: Visualization

**Segmentation:**
- `totalsegmentator`: Kidney segmentation (runs CLI externally)

**DICOM I/O:**
- `pydicom`, `python-gdcm`: DICOM reader support

**Utilities:**
- `tqdm`, `pyyaml`, `requests`: Utilities and network

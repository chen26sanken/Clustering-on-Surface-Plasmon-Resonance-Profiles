# Deciphering the Hit Patterns on SPR Sensorgrams of Small Molecules Targeting CAG Repeat DNA

This repository presents an analysis of hit patterns on SPR sensorgrams for small molecules that target CAG repeat DNA. The main objective is to cluster and categorize these hit patterns for better understanding and prediction.

## Overview

### The Concept of SPR Assay

Surface Plasmon Resonance (SPR) is a powerful tool for analyzing molecular interactions in real time. The assay provides insights into reaction dynamics, making it invaluable in molecular biology and pharmacology.

![SPR Image](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/856c0589-8f9b-4603-9073-f7d060f342aa)

**Fig. 1:** Graphical illustration of this study. (A) Target nucleic acids are immobilized on a gold-glass interface. Small molecules will flow through the immobilized targets to reach the binding equivalence in phase 1 and be washed out from the bound state in phase 2. The corresponding association and dissociation constants are denoted as Kon and Koff, respectively.

### Clustering SPR Profiles

Through clustering techniques, we can group similar SPR profiles, revealing patterns and trends that might be missed in individual analysis.

![Clustering Image](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/28bb41e7-70f9-4806-8edc-1418d620575c)

**Fig. 2:** The observed SPR sensorgrams which include the binding kinetic features are grouped by clustering algorithms into different patterns. Only hit patterns are shown here.

## Installation Instructions

To reproduce the analysis and visualizations in this repository, you'll need to install some software packages. The list below provides the names and versions required:

- `matplotlib` ~3.5.3
- `pandas` ~1.5.1
- `scikit-learn` ~1.1.3
- `numpy` ~1.23.4
- `tslearn` ~0.5.2
- `seaborn` ~0.12.2
- `scipy` ~1.9.3
- `pypdf2` ~3.0.1

To install all these packages at once, run:

```bash
pip install matplotlib~=3.5.3 pandas~=1.5.1 scikit-learn~=1.1.3 numpy~=1.23.4 tslearn~=0.5.2 seaborn~=0.12.2 scipy~=1.9.3 pypdf2~=3.0.1

# Deciphering the Hit Patterns on SPR Sensorgrams of Small Molecules Targeting CAG Repeat DNA

This repository presents an analysis of hit patterns on SPR sensorgrams for small molecules that target CAG repeat DNA. The main objective is to cluster and categorize these hit patterns for a better understanding and prediction.

## Overview

### The Concept of SPR Assay

Surface Plasmon Resonance (SPR) is a powerful tool for analyzing molecular interactions in real time. The assay provides insights into reaction dynamics, making it invaluable in molecular biology and pharmacology.

![SPR](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/fa70fad9-b79c-40a2-8a8e-7257cd98b50e)
**Fig. 1:** Graphical illustration of this study. (A) Target nucleic acids are immobilized on a gold-glass interface. Small molecules flow through the immobilized targets, binding in phase 1 and washing out in phase 2. Kon and Koff represent the association and dissociation constants, respectively.

### Clustering SPR Profiles

Clustering techniques group similar SPR profiles, revealing patterns and trends that might be missed with individual analysis.

![Clustering Image](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/28bb41e7-70f9-4806-8edc-1418d620575c)
**Fig. 2:** The observed SPR sensorgrams, which include binding kinetic features, are grouped by clustering algorithms into different patterns. Only hit patterns are displayed here.

## Code and Results 

The code accompanying this manuscript processes raw SPR data, applies clustering algorithms, and visualizes the results. The outcomes are a series of clustered hit patterns that showcase various binding behaviors of small molecules to the CAG repeat DNA.

![clustering_hits](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/78b409e9-1c26-402c-b80a-38dc46107013)
**Fig. 3:** SPR profiles in hit clusters. The different clusters represent varying kinetic behaviors of small molecules binding to CAG repeat DNA. Detailed insights from the results can be found in the accompanying manuscript.

<br>

![cluster_visualization](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/b9144f87-dd44-4482-a29b-5c32ac048425)
**Fig. 4:** Visualization of hit clusters and their four characteristic patterns. (A) UMAP, (B) t-SNE, and (C) PCA analysis. The represented cluster colors are consisted with Fig. 3.

## Installation Instructions

To reproduce the analysis and visualizations, install the required software packages:

- `matplotlib` ~3.5.3
- `pandas` ~1.5.1
- `scikit-learn` ~1.1.3
- `numpy` ~1.23.4
- `tslearn` ~0.5.2
- `seaborn` ~0.12.2
- `scipy` ~1.9.3
- `pypdf2` ~3.0.1

Use this command:

```bash
pip install matplotlib~=3.5.3 pandas~=1.5.1 scikit-learn~=1.1.3 numpy~=1.23.4 tslearn~=0.5.2 seaborn~=0.12.2 scipy~=1.9.3 pypdf2~=3.0.1

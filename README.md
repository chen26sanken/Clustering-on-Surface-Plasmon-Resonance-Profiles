# Deciphering the Hit Patterns on SPR Sensorgrams of Small Molecules Targeting CAG Repeat DNA

This repository presents an analysis of hit patterns on SPR sensorgrams for small molecules that target CAG repeat DNA. 
The study developed a clustering-based approach for grouping surface plasmon resonance (SPR) sensorgrams based on the kinetic features to identify potential hit compounds in screening. Our method successfully distinguished 220 hits with four patterns targeting CAG repeat DNA from all 2000 samples. 

## Overview

### The Concept of SPR Assay and clustering on SPR Profiles

Surface Plasmon Resonance (SPR) is a powerful tool for analyzing molecular interactions in real time. The assay provides insights into reaction dynamics, making it invaluable in molecular biology and pharmacology. Clustering techniques group similar SPR profiles, revealing patterns and trends that might be missed with individual analysis.

![figure1_SPR_clustering](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/36b4b194-a66b-466c-9c4c-ca6f96395a0d)
**Fig. 1:** Graphical illustration of this study. (A) Target nucleic acids are immobilized on a gold-glass interface. Small molecules flow through the immobilized targets, binding in phase 1 and washing out in phase 2. Kon and Koff represent the association and dissociation constants, respectively. (B) The observed SPR sensorgrams, which include binding kinetic features, are grouped by clustering algorithms into different patterns. Only hit patterns are displayed here.


## Code and Results 

The code accompanying this manuscript processes raw SPR data, applies clustering algorithms, and visualizes the results. The outcomes are a series of clustered hit patterns that showcase various binding behaviors of small molecules to the CAG repeat DNA.

![clustering_hits](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/78b409e9-1c26-402c-b80a-38dc46107013)
**Fig. 2:** SPR profiles in hit clusters. The different clusters represent varying kinetic behaviors of small molecules binding to CAG repeat DNA. Detailed insights from the results can be found in the accompanying manuscript.

<br>

![cluster_visualization](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/b9144f87-dd44-4482-a29b-5c32ac048425)
**Fig. 3:** Visualization of hit clusters and their four characteristic patterns. (A) UMAP, (B) t-SNE, and (C) PCA analysis. The represented cluster colors are consisted with Fig. 2 above.

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

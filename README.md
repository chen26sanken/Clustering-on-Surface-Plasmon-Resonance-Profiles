# Clustering on Surface Plasmon Resonance Profiles to Improve Hit Assignment of Small Molecules Targeting CAG Repeat DNA

This repository presents an analysis of hit patterns on SPR sensorgrams for 2000 small molecules that target CAG repeat DNA. Please find the paper with a same title.

## Overview

### The Concept of SPR Assay and clustering on SPR Profiles
![Github](https://github.com/user-attachments/assets/d9f714ac-5656-4efb-8232-e4e2479b91e1)



**Fig. 1:** Graphical illustration of this study.

<br>
<br>

## Code and Results 

The code accompanying this manuscript processes raw SPR data, applies clustering algorithms, and visualizes the results. The outcomes are a series of clustered hit patterns that showcase various binding behaviors of small molecules to the CAG repeat DNA. 
> **Note:** For access to the SPR data, please contact the author directly if necessary.

***



![clustering_hits](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/78b409e9-1c26-402c-b80a-38dc46107013)
**Fig. 2:** SPR profiles in hit clusters. The different clusters represent varying kinetic behaviors of small molecules binding to CAG repeat DNA. Detailed insights from the results can be found in the accompanying manuscript.

<br>

***

![cluster_visualization](https://github.com/chen26sanken/Clustering_with_SPR_profiles/assets/141697122/b9144f87-dd44-4482-a29b-5c32ac048425)
**Fig. 3:** Visualization of hit clusters and their four characteristic patterns. (A) UMAP, (B) t-SNE, and (C) PCA analysis. The represented cluster colors are consisted with Fig. 2 above.



<br>
<br>

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

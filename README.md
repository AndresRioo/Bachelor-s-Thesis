# Bachelor's Thesis: Cluster-Guided Margin Learning for Few-Shot Classification

## Overview

Few-Shot Learning (FSL) aims to build machine learning models capable of recognizing new categories from only a handful of labeled examples. While recent approaches have achieved promising results by incorporating semantic information, classification errors often remain concentrated between visually and semantically similar classes.

This project investigates the relationship between semantic structure and embedding space geometry in semantic-aware Few-Shot Learning. The work is based on the SemFew framework and explores several strategies designed to improve class separation by modifying how embeddings are organized during training.

The project was developed as part of the Bachelor's Degree in Computer Engineering at the University of Barcelona.

---

## Motivation

Traditional deep learning models typically require large amounts of labeled data to achieve strong performance. However, in many real-world scenarios obtaining sufficient annotations is expensive, impractical, or impossible.

Few-Shot Learning addresses this challenge by enabling models to generalize to unseen classes using only a few examples. Among existing approaches, methods that incorporate semantic knowledge have shown particular promise because they complement limited visual information with external semantic representations.

During the analysis of SemFew, it was observed that a significant portion of classification errors occurred between categories that were both visually and semantically similar. This observation motivated the exploration of new regularization strategies aimed at restructuring the embedding space.

---

## Objectives

The main objective of this research is to analyze the limitations of semantic-aware Few-Shot Learning methods and explore potential improvements through embedding space regularization.

Specific goals include:

- Study the state of the art in Few-Shot Learning.
- Analyze the SemFew framework and identify potential weaknesses.
- Investigate the role of semantic similarity in classification errors.
- Design novel regularization strategies based on semantic relationships between classes.
- Evaluate the impact of these strategies under a standardized experimental setup.
- Analyze how embedding geometry evolves under different regularization mechanisms.

---

## Methodology

### 1. Literature Review

The project begins with a comprehensive review of:

- Deep Learning for image classification.
- Metric learning and embedding spaces.
- Transfer Learning and pretrained visual backbones.
- Few-Shot Learning methodologies.
- Semantic-aware Few-Shot Learning approaches.

Special attention is given to the SemFew architecture, which serves as the baseline model throughout the study.

---

### 2. Problem Analysis

An in-depth analysis of SemFew is conducted to identify common failure patterns.

The study focuses on:

- Semantic relationships between classes.
- Distribution of embeddings in feature space.
- Confusion patterns among similar categories.
- Limitations of existing prototype construction mechanisms.

This analysis serves as the foundation for formulating new research hypotheses.

---

### 3. Proposed Approaches

Three independent modifications are developed and evaluated.

#### Adaptive Semantic Margin

Introduces margins whose magnitude depends on the semantic similarity between classes.

The intuition is that categories sharing strong semantic relationships may require additional separation within the embedding space to improve discrimination.

#### Hierarchical Clustering with Fixed Radius

Applies hierarchical clustering to identify groups of semantically related categories.

Cluster centers are then used to define exclusion regions that encourage embeddings to avoid highly ambiguous areas of the representation space.

#### Hierarchical Clustering with Adaptive Radius

Extends the previous approach by allowing exclusion regions to adapt dynamically according to the internal structure of each semantic cluster.

This method aims to provide a more flexible regularization mechanism that better reflects semantic variability across categories.

---

## Experimental Setup

Experiments are conducted using:

- SemFew as the baseline framework.
- CIFAR-FS benchmark dataset.
- Episodic Few-Shot evaluation protocols.
- Standard N-way K-shot classification settings.

In addition to final classification accuracy, several internal metrics are analyzed to study how each proposal affects the geometric structure of the learned embeddings.

---

## Research Contributions

This work contributes:

- A detailed analysis of error patterns in semantic-aware Few-Shot Learning.
- Three novel embedding-space regularization strategies.
- An experimental study of semantic clustering for prototype-based classification.
- A geometric analysis framework for understanding embedding organization beyond accuracy metrics.

---


## Academic Information

**Author:** Andrés Rio Nogues  
**Degree:** Bachelor's Degree in Computer Engineering  
**University:** University of Barcelona  
**Faculty:** Faculty of Mathematics and Computer Science  
**Year:** 2026

---

## Citation

If you use this work for academic purposes, please cite the repository and the original SemFew paper on which this research is based.

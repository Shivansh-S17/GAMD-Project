# 🌐 GADM: Geometry-Acquainted Deep Model for 3D Point Cloud Analysis

This repository implements **GADM**, a deep learning framework that captures both **local and global geometric features** of 3D point clouds. The model is designed for both **classification** and **segmentation** tasks using custom-designed neural network architectures with geometry-aware feature extraction.

---

## 📌 Summary

3D point cloud analysis remains challenging due to its **irregular** and **unordered** nature. Existing models often use convolutional, graph-based, or attention mechanisms to exploit spatial relationships, but they typically ignore a comprehensive fusion of **local and global geometry**.

In this project, we introduce:
- A novel **Geometry-Acquainted Fusion (GAF)** module
- Feature encoding using **in-plane** and **out-plane distances**
- Two separate architectures:
  - A **Feed-Forward Network** for Classification
  - A **U-Net-like Residual Network** for Segmentation

GADM is evaluated against state-of-the-art models and demonstrates superior or comparable performance across standard benchmarks.

---

---

## 🧠 Key Components

### 🔹 Geometry-Acquainted Fusion (GAF) Module
Captures global-to-local geometry across multiple abstraction levels using:
- **Multi-step fusion**
- **In-plane & out-plane distance metrics**
- **Learned embeddings** for raw spatial structure

### 🔹 Classification Network
- Simple **feed-forward architecture**
- Uses GAF features directly for classification
- Lightweight and high-speed

### 🔹 Segmentation Network
- **U-Net-like** architecture
- Enhanced with **residual connections**
- Outputs point-wise class labels

---




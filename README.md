# 3DPointCloud-Notes
This is the collection of my notes on the online course 3D pointcloud processing in Shenlanxueyuan.

## Content
* [1. Introduction](#1-introduction)
* [2. Image and Video Compressing](#2-image-and-video-compressing)
* [3. Spatial Processing](#3-spatial-processing)
* [4. Image Restoration](#4-image-restoration)
* [5. Image Segmentation](#5-image-segmentation)
* [6. Geometric PDEs](#6-geometric-pDEs)
* [7. Image and Video Inpainting](#7-image-and-video-inpainting)
* [8. Sparse Modeling and Compressed Sensing](#8-sparse-modeling-and-compressed-sensing)
* [9. Medical Imaging](#9-medical-imaging)


## 1. Introduction
   * **3D point Cloud**
        * Representation: N x (3+D) 
        * Data Source:
            * Lidar
            * RGB-D
            * CAD
   * 3D description
        * mesh
        * Voxel grid: 不高效，因为稀疏性。
        * octree
   * Difficulties
        * Sparsity (far)
        * Irregular (compared to pixel) in neighbor searching 
        * No texture information  (three people and a car)
        * Un-order (in matrix) - difficult in deep learning
        * Rotation invariance
   * **PCA (linear)** - find the dominant direction
       * Application:
            * Dimensionality reduction
            * Surface normal
            * Classification 
       * SVD
       * Spectrum theorem
       * Rayleigh Quotient 
       * **Theory:**
           * Input: data
           * Output: principle vectors (large variance)
       * Process:
           * Normalization
           * Calculate variance with projection using matrix form
           * Maximize variance to find z1=u1
           * Find z2 by deflation (remove z1 effect), which leads to z1=u2
           * note:  eigen values should be in order!
   * **Kernel PCA (nonlinear)** - lift dimension 
       * Process:
           * Find extra dimension
           * PCA
       * Problem
           * How to determine phi()
           * How to deal with high dimension
       * 见笔记
   * **Surface Normal**
       * PCA:
           * Normal - least significant vector
           * Curvature - lambda3/(lambda3+lambda2+lambda1)
       * Weighted normal estimation: 利用相似度决定权重
       * Noise removal
           * Radius
           * Statistical 
       * Downsampling
           * Voxel grid
               * Hash table
       * Farthest point sampling (FPS)
       * Normal space sampling (NSS): 
           * 法向量角度考虑均匀分布
           * 用于ICP
       * Upsampling 
           * bilateral filter - gaussian
               * Edge preserving - 利用颜色相似度

---

## Why Switch from an MLP to a CNN? (Engineering Post-Mortem)

After training our two-layer NumPy Multilayer Perceptron (MLP) for 7,000 iterations and achieving ~90% training accuracy, empirical evaluation on unseen test images exposed a severe generalization bottleneck: real-world accuracy dropped to **40% (4/10)**.

### 1. The Core Limitation: Spatial Oblivion
* **Flattening Destroys Geometry**: An MLP flattens a $150 \times 150 \times 3$ RGB image into a flat vector of $67,500$ independent numbers. The network has no geometric awareness that pixel $(x, y)$ is physically adjacent to pixel $(x+1, y)$.
* **Shortcut Memorization**: The model relied on global color histograms and background pixels rather than flower anatomy. For instance, a dandelion in front of a blue sky was classified as a **Bellflower with 99.0% confidence** because the model associated high blue pixel values at the top coordinates with bellflower backgrounds.
* **Lack of Translation Invariance**: If a flower shifts 10 pixels to the right, every input coordinate shifts to entirely different weight connections in an MLP.

### 2. The Architectural Solution: Convolutional Neural Networks (CNNs)
A CNN decouples the task into two sequential stages:
1. **Feature Extraction (The Visual Cortex)**:
   * **Convolutional Layers (`Conv2d`)**: Slide small $3 \times 3$ parameter-sharing kernels across the spatial grid to detect local features (edges, curves, textures) regardless of where they appear in the frame.
   * **Max Pooling (`MaxPool2d`)**: Progressively downsamples spatial dimensions ($150 \to 75 \to 37 \to 18$), shedding redundant pixel noise, preserving only the strongest feature activations, and enforcing translation invariance.
2. **Classification (The Decision Maker)**:
   * Only after the image is compressed into high-level, meaningful feature representations (e.g., "radiating petal patterns") does a dense linear layer classify the flower.

### 3. Transitioning to PyTorch
While implementing an MLP in pure NumPy provided deep mastery of forward/backward propagation and gradient descent mechanics, building a CNN from scratch in NumPy is computationally impractical on CPU. PyTorch provides:
* **CUDA Hardware Acceleration**: Offloading 2D tensor operations to the RTX 5050 GPU (8 GB VRAM).
* **Dynamic Autograd**: Eliminating manual chain-rule calculus for complex layers like 2D convolutions and Batch Normalization.

# TC-NO: Topologically Conditioned Neural Operator

## 🚀 Overview
**TC-NO (Topologically Conditioned Neural Operator)** is a deep learning architecture designed for financial time-series forecasting. It bridges **Topological Data Analysis (TDA)** and **Fourier Neural Operators (FNO)** to capture complex, non-linear dependencies in asset markets while enforcing financial constraints.

This repository contains a self-contained, demonstrable PyTorch implementation of the TC-NO architecture. It simulates the extraction of topological market features (Persistent Homology) and uses them to dynamically condition a Neural Operator to predict future asset returns.

## 🧠 Core Concepts

The model addresses three key challenges in financial modeling:
1.  **Market Structure:** Captures shifting correlations and regimes using Topology (TDA).
2.  **Resolution Invariance:** Uses Neural Operators to learn the underlying continuous dynamic system, rather than just pixelated discrete steps.
3.  **Financial Theory:** Enforces a "No-Arbitrage" constraint using a geometric loss function.

## 🏗️ Architecture Modules

The code is organized into three distinct modules:

### 1. Topological Feature Encoder (TFE)
*   **Goal:** Convert a rolling window of asset returns into a latent topological vector $z_t$.
*   **Method:** Computes a correlation distance matrix $D_{ij} = \sqrt{2(1 - \rho_{ij})}$. It simulates the filtration process of Persistent Homology to extract features representing market loops and clusters.
*   **Code Reference:** `class TopologicalFeatureEncoder`

### 2. Neural Operator Core (NOC)
*   **Goal:** Map historical states to future return functions.
*   **Method:** A Fourier Neural Operator (FNO) where the spectral filter weights are **dynamically modulated** by the topological vector $z_t$. This allows the model to change its frequency response based on the current market regime (e.g., Crisis vs. Stable).
*   **Code Reference:** `class FNOBlock`, `class SpectralFilterModulator`

### 3. Constraint Enforcement Module (CEM)
*   **Goal:** Ensure predictions adhere to the Efficient Market Hypothesis.
*   **Method:** A geometric **No-Arbitrage Loss**. It identifies triangular arbitrage loops (cycles in the correlation graph) and penalizes the model if it predicts a risk-free profit loop within those cycles.
*   **Code Reference:** `function NoArbitrageLoss`

## 📦 Dependencies

To run this code, you need Python installed with the following libraries:

```bash
pip install torch numpy pandas matplotlib ipython
```

## 🏃 Usage

The provided script is a self-contained demonstration. It includes mock data generation, model initialization, a forward pass/training step, and visualization.

Simply run the script in a Jupyter Notebook or as a standard Python file:

```python
# If running as a script
python tcno_demo.py
```

### Expected Output
1.  **Training Summary:** A printout showing the Total Loss, MSE, and Arbitrage Violation penalty for a simulated training step.
2.  **Performance Table:** A Pandas DataFrame comparing TC-NO against baselines (Random Walk, LSTM, etc.).
3.  **Visualization:** A `matplotlib` figure illustrating:
    *   **Persistence Diagrams:** How market topology changes across regimes.
    *   **Spectral Kernels:** How the FNO adapts its frequency focus based on the topology.

## 📝 Code Implementation Details

### The "Mock" Components
*   **TDA Simulation:** Real TDA requires libraries like GUDHI or Dionysus. To keep this script standalone and lightweight, the `identify_arbitrage_loops` function **simulates** the output of a Persistent Homology H1 generator by finding the "tightest" correlation triangles (lowest distance loops) manually using NumPy.
*   **Data:** The script generates random tensors to simulate asset returns.

### The No-Arbitrage Logic
The loss function calculates:
$$ L_{Arb} = \mathbb{E} \left[ \sum_{C \in \mathcal{C}} | P_C(\hat{r}) | \right] $$
Where $C$ represents a cycle (loop) of assets, and $P_C$ is the return of a portfolio weighted to traverse that loop. If the market is efficient, the sum of returns around a closed loop of highly correlated assets should tend toward zero (after costs).

## 📊 Visuals Generated

The script generates a conceptual figure demonstrating the **Topology Transition**:

1.  **Regime I (Stable):** Sparse connectivity, balanced spectral response.
2.  **Regime II (Transition):** Fragmentation, high-frequency focus.
3.  **Regime III (Crisis):** Global connectivity (market crash correlation), low-frequency global smoothing.

---

**License:** MIT  
**Author:** [Your Name/Organization]

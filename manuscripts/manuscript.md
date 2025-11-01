# scQDiff: Learning Temporal Regulatory Logic and Guided Trajectories via Quantum-Inspired Schrödinger Bridge Diffusion

**Authors:**

Tommy W. Terooatea

---

## Abstract

Understanding the dynamic regulatory programs governing cell state transitions requires moving beyond static network inference to capture time-varying gene-gene interactions. We introduce **scQDiff** (single-cell Quantum Diffusion), a mathematical framework that leverages the Schrödinger Bridge problem—a quantum-mechanical approach to optimal transport—to learn continuous, interpretable regulatory dynamics from single-cell RNA sequencing data. scQDiff constructs a **temporal Jacobian tensor** 𝓙(*t*) ∈ ℝ<sup>G×G×T</sup> that encodes how regulatory influences evolve across pseudotime. Through low-rank tensor decomposition, we extract **regulatory archetypes**—fundamental dynamic patterns corresponding to distinct biological programs—each with an associated temporal activation profile. These archetypes enable **guided trajectory synthesis**, allowing both forward prediction of cellular responses and reverse engineering of reprogramming strategies. The quantum-inspired formulation provides principled handling of stochastic dynamics, non-equilibrium processes, and bidirectional inference. We present the complete mathematical framework, discuss computational strategies, and propose experimental validation designs. scQDiff establishes a rigorous foundation for mechanistic, predictive modeling of cellular decision-making with broad applications in developmental biology, immunology, and therapeutic design.

---

## Introduction

### The Challenge: From Snapshots to Dynamics

Single-cell genomics has revolutionized our ability to catalog cellular states at unprecedented resolution. However, a fundamental challenge remains: inferring the **dynamic regulatory logic** that governs transitions between these states. Existing trajectory inference methods often rely on oversimplified assumptions—such as a single global equilibrium or linear velocity fields—and lack mechanistic interpretability. While RNA velocity [4] provides directional information from splicing dynamics, it assumes a constant regulatory program and cannot capture the time-varying nature of gene-gene interactions during complex processes like differentiation, activation, or disease progression.

### The scQDiff Solution: Quantum-Inspired Dynamics

We introduce scQDiff, a framework that addresses these limitations by grounding trajectory inference in the mathematics of **Schrödinger Bridges** [1,2,3]. The Schrödinger Bridge problem seeks the most probable stochastic process connecting two probability distributions, subject to minimal deviation from a reference diffusion (typically Brownian motion). This formulation has deep connections to quantum mechanics: it involves Schrödinger potentials, the Hamilton-Jacobi equation, and the Bohm quantum potential, which introduces non-local effects analogous to quantum entanglement [1].

By adopting this quantum-inspired perspective, scQDiff models cell state transitions as probabilistic diffusion processes that:

1. **Respect stochasticity**: Biological variability is not noise but an intrinsic feature of the dynamics
2. **Capture non-equilibrium behavior**: Cells evolve through transient, time-varying regulatory states
3. **Enable bidirectional inference**: Both forward (prediction) and reverse (reprogramming) trajectories emerge naturally from the same mathematical structure
4. **Provide mechanistic insight**: The learned dynamics can be decomposed into interpretable regulatory modules

### Key Innovation: The Temporal Jacobian Tensor

The centerpiece of scQDiff is the **temporal Jacobian tensor** 𝓙 ∈ ℝ<sup>G×G×T</sup>, which captures how the regulatory network evolves over pseudotime. Each slice 𝓙(*t*) = *J*(*t*) is the Jacobian matrix of the drift field *f*(*x*, *t*) at time *t*, where *J<sub>ij</sub>*(*t*) represents the instantaneous influence of gene *j* on gene *i*. By decomposing this tensor into **regulatory archetypes**—low-rank patterns corresponding to distinct biological programs—we obtain a compact, interpretable representation of the complex dynamics.

---

## Mathematical Framework

### Notation and Problem Setup

We consider single-cell RNA-seq data represented as an expression matrix **X** ∈ ℝ<sup>n×G</sup>, where *n* is the number of cells and *G* is the number of genes (or a curated subset of regulatory genes). Each cell is assigned a pseudotime *t* ∈ [0, 1] based on trajectory inference methods or experimental time points. Our goal is to learn a continuous **drift field** *f*(*x*, *t*) : ℝ<sup>G</sup> × [0,1] → ℝ<sup>G</sup> that describes the instantaneous rate of change of gene expression.

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| **X** | Expression matrix | *n* × *G* |
| *x* | Gene expression vector for a single cell | ℝ<sup>G</sup> |
| *t* | Pseudotime | [0, 1] |
| *f*(*x*, *t*) | Drift field (velocity) | ℝ<sup>G</sup> |
| *J*(*t*) | Jacobian matrix at time *t* | ℝ<sup>G×G</sup> |
| 𝓙 | Temporal Jacobian tensor | ℝ<sup>G×G×T</sup> |
| *M<sub>k</sub>* | Regulatory archetype *k* | ℝ<sup>G×G</sup> |
| *a<sub>k</sub>*(*t*) | Temporal activation profile of archetype *k* | ℝ |

### Step 1: The Schrödinger Bridge and Drift Field Inference

#### The Quantum Connection

The Schrödinger Bridge problem [1,2,3] seeks the most probable stochastic process connecting two marginal distributions *p*<sub>start</sub> and *p*<sub>target</sub>. Mathematically, this is formulated as an entropic optimal transport problem:

$$\min_{\pi} \mathbb{E}_{\pi}[c(x_0, x_1)] + \varepsilon \, \text{KL}(\pi \| \pi_{\text{ref}})$$

where:
- *π* is a coupling (joint distribution) between start and target
- *c*(*x*₀, *x*₁) is the transport cost (typically ‖*x*₀ − *x*₁‖²)
- *ε* > 0 is the entropic regularization parameter (analogous to ℏ in quantum mechanics)
- *π*<sub>ref</sub> is a reference measure (e.g., Brownian motion)
- KL denotes the Kullback-Leibler divergence

The solution involves **Schrödinger potentials** *φ* and *ψ* that satisfy coupled nonlinear equations analogous to the time-dependent Schrödinger equation [2]:

$$\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \nabla \phi) + \varepsilon \Delta \rho$$

This formulation introduces the **Bohm quantum potential**, which captures non-local regulatory effects—perturbations at one gene can instantaneously affect the regulatory landscape globally, similar to quantum entanglement [1].

#### Drift Field Learning: Two Approaches

We provide two complementary strategies for learning the drift field *f*(*x*, *t*):

**Option 1A: Velocity-Constrained Field (Gene Space)**

When RNA velocity estimates *v̂*(*x*) are available, we learn *f<sub>θ</sub>*(*x*, *t*) by minimizing:

$$\mathcal{L}_{\text{velocity}}(\theta) = \mathbb{E}_{x,t}\left[\|f_\theta(x,t) - \hat{v}(x)\|_2^2\right] + \lambda_{\text{graph}} \sum_{i,j} W_{ij} \|f_\theta(x_i,t) - f_\theta(x_j,t)\|_2^2 + \lambda_{\text{wd}} \|\theta\|_2^2$$

The three terms enforce:
1. **Velocity matching**: Consistency with observed splicing dynamics
2. **Graph smoothness**: Similar cells (neighbors in the kNN graph with weights *W<sub>ij</sub>*) should have similar drift vectors
3. **Regularization**: Prevents overfitting

**Option 1B: Bridge-Lite (Latent Space)**

Alternatively, we learn a drift field *g<sub>φ</sub>*(*z*, *t*) in a latent space *z* = *E*(*x*) (e.g., from a variational autoencoder) using a Schrödinger Bridge / Optimal Transport objective:

$$\mathcal{L}_{\text{bridge}}(\phi) = \sum_{k=1}^{T-1} \text{KL}\left(\mathcal{T}_{g_\phi}(p(z|t_k)) \,\|\, p(z|t_{k+1})\right) + \beta \int_0^1 \mathbb{E}_{z,t}\left[\|g_\phi(z,t)\|_2^2\right] dt$$

where 𝒯<sub>*g*</sub> denotes the pushforward of the distribution under the flow induced by *g*. The first term ensures that the learned dynamics transport the marginal distribution at time *t<sub>k</sub>* to match the observed marginal at *t<sub>k+1</sub>*, while the second term (control energy) penalizes large drift magnitudes. The drift in gene space is recovered via the decoder Jacobian: *f*(*x*, *t*) = (∂*D*/∂*z*) · *g<sub>φ</sub>*(*E*(*x*), *t*).

Both approaches incorporate **biological guards**:
- Laplacian regularization (*λ*<sub>graph</sub>) keeps trajectories on the data manifold
- Bounded step sizes prevent off-manifold excursions
- Optional integration of regulatory priors (SCENIC+, PINNACLE) to constrain *f* to biologically feasible directions

### Step 2: Computing the Temporal Jacobian Tensor

Once we have learned the drift field *f*(*x*, *t*), we compute its **Jacobian matrix** at discrete time points {*t*₁, *t*₂, ..., *t<sub>T</sub>*} ⊂ [0, 1]:

$$J_{ij}(t) = \frac{\partial f_i(x,t)}{\partial x_j}$$

This matrix encodes the **local regulatory sensitivity**: *J<sub>ij</sub>*(*t*) > 0 indicates that increasing gene *j* increases the rate of change of gene *i* (activation), while *J<sub>ij</sub>*(*t*) < 0 indicates repression.

#### Option 2A: Weighted Local Ridge Regression (Fast)

For cells in a temporal window around *t<sub>k</sub>*, weighted by a Gaussian kernel *w<sub>i</sub>* = exp(−(*t<sub>i</sub>* − *t<sub>k</sub>*)² / 2*σ*²), we fit a local linear map:

$$\min_{W(t_k)} \sum_i w_i \|\hat{v}(x_i) - W(t_k) x_i\|_2^2 + \alpha \|W(t_k)\|_F^2$$

Then *J*(*t<sub>k</sub>*) ≈ *W*(*t<sub>k</sub>*). The ridge penalty *α* ensures numerical stability.

#### Option 2B: Autograd Jacobian (Exact)

If *f<sub>θ</sub>* is a differentiable neural network, we compute the exact Jacobian via automatic differentiation:

$$J_{ij}(t_k) = \frac{1}{|\mathcal{N}(t_k)|} \sum_{x \in \mathcal{N}(t_k)} \frac{\partial f_i(x, t_k)}{\partial x_j}$$

where 𝒩(*t<sub>k</sub>*) is the set of cells near *t<sub>k</sub>*. This is efficiently computed using vector-Jacobian products (VJPs) or Jacobian-vector products (JVPs).

#### Assembling the Tensor

We stack the Jacobian slices to form the **temporal Jacobian tensor**:

$$\boxed{\mathcal{J}_{i,j,k} = J(t_k)_{ij}} \quad \in \mathbb{R}^{G \times G \times T}$$

This tensor is the central object of scQDiff, encoding the complete time-varying regulatory network.

### Step 3: Regulatory Archetype Extraction via Tensor Decomposition

The raw tensor 𝓙 is high-dimensional and difficult to interpret directly. We seek a **low-rank decomposition** that reveals the fundamental regulatory patterns:

$$\mathcal{J}(t) \approx \sum_{k=1}^{K} a_k(t) M_k$$

where:
- *M<sub>k</sub>* ∈ ℝ<sup>G×G</sup> is the *k*-th **regulatory archetype** (a gene×gene influence matrix)
- *a<sub>k</sub>*(*t*) ∈ ℝ is the **temporal activation profile** (when archetype *k* is active)
- *K* ≪ *T* is the number of archetypes (typically 3–6)

#### Singular Value Decomposition (SVD) Algorithm

1. **Reshape** the tensor: 𝓙 → **M** ∈ ℝ<sup>T × (G·G)</sup> (time × flattened gene-gene matrix)
2. **Compute SVD**: **M** = **U Σ V**<sup>T</sup>
3. **Extract components**:
   - Temporal profiles: *a<sub>k</sub>*(*t*) = **U**<sub>:,*k*</sub> · *Σ<sub>kk</sub>* (the *k*-th left singular vector, scaled by singular value)
   - Regulatory archetypes: *M<sub>k</sub>* = reshape(**V**<sub>:,*k*</sub>, [*G*, *G*]) (the *k*-th right singular vector, reshaped to gene×gene matrix)

The singular values *Σ<sub>kk</sub>* indicate the importance of each archetype. We select *K* such that the top *K* components explain ≥ 90% of the variance.

#### Biological Interpretation of Archetypes

Each archetype *M<sub>k</sub>* has a specific biological meaning:

| Archetype Property | Interpretation |
|-------------------|----------------|
| **Rows with large magnitude** | Genes whose expression is strongly regulated by this archetype |
| **Columns with large magnitude** | Genes that act as regulators (e.g., transcription factors) |
| **Positive entries** *M<sub>k</sub>*<sub>*ij*</sub> > 0 | Gene *j* activates gene *i* |
| **Negative entries** *M<sub>k</sub>*<sub>*ij*</sub> < 0 | Gene *j* represses gene *i* |
| **Temporal profile** *a<sub>k</sub>*(*t*) | When the archetype is active (e.g., early, mid, late in the process) |

For example, in a hypothetical T-cell activation scenario:
- **M₁**: Early TCR signaling (high *a*₁(*t*) at *t* ≈ 0.1–0.3), with TFs like *NFAT*, *AP1* activating immediate-early genes
- **M₂**: Mid-transition metabolic reprogramming (high *a*₂(*t*) at *t* ≈ 0.4–0.6), with *MYC*, *HIF1A* driving glycolysis
- **M₃**: Late effector function (high *a*₃(*t*) at *t* ≈ 0.7–1.0), with *TBX21*, *EOMES* activating cytokine production

![Figure 2: Decomposition of Jacobian tensor into underlying regulatory archetypes.](figures/figure_jacobian_decomposition.png)
*Figure 2: Decomposition of the temporal Jacobian tensor into regulatory archetypes. The 3D tensor (genes × genes × time) is factorized via SVD into a set of archetypes {M₁, M₂, M₃} and their temporal activation profiles {a₁(t), a₂(t), a₃(t)}. Each archetype represents a distinct regulatory module active at specific stages of the biological process.*

### Step 4: Guided Trajectory Synthesis

A key innovation of scQDiff is using the learned archetypes to **guide** synthetic cell trajectories. This enables both **forward prediction** (what will happen) and **reverse engineering** (how to reprogram).

#### Forward Synthesis: Predicting Cellular Responses

Given a starting cell state *x*₀ (e.g., naive cells), we wish to predict its trajectory toward a target state (e.g., activated cells). We modify the base drift field with **archetype-based guidance**:

$$\tilde{f}(x,t) = f(x,t) + \gamma \sum_{k=1}^{K} a_k(t) v_k(x)$$

where:
- *f*(*x*, *t*) is the baseline drift (learned in Step 1)
- *v<sub>k</sub>*(*x*) is the vector field induced by archetype *M<sub>k</sub>* (computed as *M<sub>k</sub> x* or via decoder projection if in latent space)
- *γ* > 0 is the **guidance strength** (typically 0.5–1.5)

We then sample a **stochastic differential equation (SDE)** trajectory:

$$dx_t = \tilde{f}(x_t, t) \, dt + \sqrt{2\varepsilon} \, dW_t$$

where *W<sub>t</sub>* is a Wiener process (Brownian motion) and *ε* is the diffusion coefficient (related to the entropic regularization in the Schrödinger Bridge).

Alternatively, for a deterministic prediction, we solve the **ordinary differential equation (ODE)**:

$$\frac{dx}{dt} = \tilde{f}(x,t)$$

#### Reverse Synthesis: Therapeutic Reprogramming

To design interventions that reverse a disease state, we run the **inverse bridge**. Given a diseased cell state *x*₁, we seek a trajectory back to a healthy state *x*₀. We use the **time-reversed** archetypes:

$$\tilde{f}_{\text{reverse}}(x,t) = f(x, 1-t) - \gamma \sum_{k=1}^{K} a_k(1-t) v_k(x)$$

The control energy required for this reversal is:

$$E_{\text{control}} = \int_0^1 \|\gamma \sum_k a_k(t) v_k(x_t)\|_2^2 \, dt$$

High control energy indicates **irreversible transitions** (e.g., senescence, terminal differentiation), while low energy suggests the process can be reversed with modest interventions.

#### Round-Trip Error: Measuring Irreversibility

To quantify how well a process can be reversed, we compute the **round-trip error**:

$$\text{RTE} = \|x_0 - \text{reverse}(\text{forward}(x_0))\|_2$$

where forward(*x*₀) generates a trajectory from *x*₀ to *x*₁, and reverse(*x*₁) attempts to return to *x*₀. Large RTE indicates irreversible changes in the regulatory network.

![Figure 1: The scQDiff workflow.](figures/scqdiff_scheme.png)
*Figure 1: The complete scQDiff workflow. (A) Single-cell RNA-seq data with pseudotime ordering is used to learn a drift field f(x,t) via Schrödinger Bridge optimization. (B) The temporal Jacobian tensor 𝓙 is computed by differentiating the drift field at discrete time points. (C) Tensor decomposition extracts regulatory archetypes and their activation profiles. (D) These archetypes guide synthetic trajectories for both forward prediction and reverse reprogramming.*

---

## Computational Considerations

### Scalability and Efficiency

The computational bottleneck of scQDiff is the Jacobian computation (Step 2). For *G* genes and *T* time points, the tensor has *G*² × *T* entries. We recommend:

1. **Gene selection**: Focus on 50–300 curated genes (TFs, signaling molecules, markers) rather than the full transcriptome
2. **Batched autograd**: Compute Jacobians for multiple time points in parallel using GPU acceleration
3. **Sparse approximations**: Leverage sparsity in regulatory networks (most gene pairs do not interact)

### Parameter Selection

| Parameter | Recommended Range | Selection Strategy |
|-----------|------------------|-------------------|
| Time points *T* | 10–50 | Balance temporal resolution vs. computational cost |
| Window bandwidth *σ* | 0.05–0.15 | 5–15% of pseudotime range; validate via cross-validation |
| Ridge penalty *α* | 10⁻³–10⁻¹ | Choose to stabilize local regression without over-smoothing |
| Number of archetypes *K* | 3–6 | Elbow plot of singular values; aim for ≥90% variance explained |
| Guidance strength *γ* | 0.5–1.5 | Tune to balance archetype influence and manifold fidelity |
| Entropic regularization *ε* | 0.001–0.1 | Smaller = more deterministic; larger = more stochastic |

### Validation and Uncertainty Quantification

To ensure robustness, we recommend:

1. **Bootstrap resampling**: Resample cells with replacement, retrain, and report mean ± confidence intervals for 𝓙<sub>*ij*</sub>(*t*) and *a<sub>k</sub>*(*t*)
2. **Ensemble models**: Train with multiple random seeds and aggregate predictions
3. **Out-of-distribution (OOD) detection**: Flag synthetic cells with high kNN distance or score norm |∇<sub>*x*</sub> log *p*(*x*)| as low-confidence
4. **Biological validation**: Compare archetypes to known TF-target databases (SCENIC+, DoRothEA); verify that chromatin accessibility (scATAC-seq) correlates with archetype activation

---

## Discussion

### Conceptual Advances

scQDiff introduces several conceptual innovations for single-cell trajectory analysis:

1. **Quantum-Inspired Dynamics**: By grounding the framework in Schrödinger Bridges, we provide a principled mathematical foundation for modeling stochastic, non-equilibrium biological processes. The quantum potential captures non-local regulatory effects, reflecting the reality that gene regulatory networks exhibit long-range dependencies.

2. **Temporal Regulatory Networks**: Unlike static network inference methods (e.g., SCENIC+), scQDiff learns how gene-gene interactions evolve over time. This is critical for processes like differentiation, where early TFs activate mid-stage signaling pathways, which in turn activate late effector programs.

3. **Interpretable Archetypes**: The low-rank decomposition distills the high-dimensional tensor into a small number of biologically meaningful patterns. Each archetype corresponds to a coherent regulatory module, making the model interpretable and testable.

4. **Bidirectional Inference**: The same mathematical structure supports both forward prediction and reverse engineering. This duality is a natural consequence of the time-symmetric Schrödinger Bridge formulation.

### Comparison to Existing Methods

| Method | Approach | Strengths | Limitations | scQDiff Adds |
|--------|----------|-----------|-------------|--------------|
| **RNA Velocity** [4] | Splicing dynamics | Fast, widely adopted | Assumes constant equilibrium; no time-varying network | Quantum bridge with temporal Jacobian |
| **CellRank** [5] | Markov chain on velocity kernel | Fate probabilities | Linear velocity assumption | Non-linear dynamics, regulatory archetypes |
| **Dynamo** [6] | ODE fitting to velocity | Continuous vector field | Equilibrium-focused | Non-equilibrium bridge, explicit OT |
| **SCENIC+** [7] | Static TF-target inference | Comprehensive regulons | No dynamics | Temporal tensor, time-varying archetypes |
| **Cflows** [8] | Neural ODE + OT | Temporal GRN inference | Requires time-series data | Works on pseudotime, quantum formulation |

### Limitations and Future Work

While scQDiff represents a significant advance, several challenges remain:

1. **Pseudotime Dependency**: The method currently requires a reasonable pseudotime ordering. Future work will explore de novo trajectory discovery by learning the Schrödinger Bridge directly from unordered cells.

2. **Computational Cost**: Jacobian estimation for large gene sets is expensive. We are developing sparse approximations and GPU-accelerated implementations.

3. **Multi-Omics Integration**: The current framework focuses on scRNA-seq. Future extensions will integrate scATAC-seq (chromatin accessibility), proteomics (post-transcriptional regulation), and spatial transcriptomics (location-dependent dynamics).

4. **Causal Validation**: While the Jacobian encodes regulatory influences, establishing true causality requires interventional experiments. We propose specific validation designs in the supplementary Future Directions section.

### Broader Impact

scQDiff has the potential to transform how we study cellular decision-making. By providing a mechanistic, predictive framework, it enables:

- **Drug discovery**: Predict cellular responses to perturbations and identify combination therapies
- **Regenerative medicine**: Design reprogramming strategies to convert diseased cells to healthy states
- **Precision medicine**: Personalize treatment by conditioning the bridge on patient-specific features
- **Fundamental biology**: Dissect the regulatory logic of development, immunity, and disease

---

## Methods Summary

A detailed mathematical derivation and computational implementation guide is provided in the supplementary documentation. Briefly, the scQDiff pipeline consists of:

1. **Data preprocessing**: Quality control, normalization, pseudotime inference, kNN graph construction
2. **Drift field learning**: Velocity-constrained or bridge-lite optimization (Step 1)
3. **Jacobian computation**: Local ridge regression or autograd (Step 2)
4. **Tensor assembly**: Stack Jacobian slices across time (Step 2)
5. **Archetype extraction**: SVD-based decomposition (Step 3)
6. **Biological validation**: Regulon alignment, ATAC coherence, bootstrap uncertainty (Step 3)
7. **Guided synthesis**: Forward prediction or reverse reprogramming (Step 4)
8. **Iterative refinement**: Optional EM-style loop to improve fit (Step 4)

---

## Conclusion

scQDiff establishes a rigorous mathematical foundation for learning and manipulating cellular regulatory dynamics. By leveraging quantum-inspired Schrödinger Bridges, we provide a principled framework for modeling stochastic, time-varying gene regulatory networks. The extraction of regulatory archetypes offers unprecedented interpretability, while guided trajectory synthesis enables both predictive modeling and therapeutic design. We anticipate that scQDiff, once validated through the experimental designs outlined in the supplementary materials, will become a powerful tool for dissecting the regulatory logic of life.

---

## References

[1] Bordyuh, M., Clevert, D., & Bertolini, M. (2025). Exact Solutions to the Quantum Schrödinger Bridge Problem. *arXiv preprint arXiv:2509.25980*. https://arxiv.org/abs/2509.25980

[2] Nutz, M., & Wiesel, J. (2022). Entropic optimal transport: Convergence of potentials. *Probability Theory and Related Fields*, 184, 401–424. https://link.springer.com/article/10.1007/s00440-021-01096-8

[3] Pavon, M. (2002). Quantum Schrödinger bridges. In *Directions in Mathematical Systems Theory and Optimization* (pp. 227–238). Springer. https://link.springer.com/chapter/10.1007/3-540-36106-5_17

[4] La Manno, G., Soldatov, R., Zeisel, A., et al. (2018). RNA velocity of single cells. *Nature*, 560, 494–498. https://www.nature.com/articles/s41586-018-0414-6

[5] Lange, M., Bergen, V., Klein, M., et al. (2022). CellRank for directed single-cell fate mapping. *Nature Methods*, 19, 159–170. https://www.nature.com/articles/s41592-021-01346-6

[6] Qiu, X., Zhang, Y., Sosina, O.A., et al. (2022). Mapping transcriptomic vector fields of single cells. *Cell*, 185, 690–711. https://www.cell.com/cell/fulltext/S0092-8674(21)01586-1

[7] Bravo González-Blas, C., De Winter, S., Hulselmans, G., et al. (2023). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks. *Nature Methods*, 20, 1355–1367. https://www.nature.com/articles/s41592-023-01938-4

[8] Tong, A., Kuchroo, M., Huang, J., et al. (2024). Revealing dynamic temporal regulatory networks driving cancer cell state plasticity with neural ODE-based optimal transport. *bioRxiv*. https://www.biorxiv.org/content/10.1101/2023.11.20.567883v2

---

## Acknowledgements

The authors acknowledge the use of AI tools (including Manus, ChatGPT, and Deepseek) for assistance in organizing the manuscript, suggesting methodological approaches, and proofreading the text.

## Author Contributions

T.W.T.: Conceptualization, Project Administration, Methodology, Writing – Original Draft Preparation, Writing – Review & Editing.

AI Tools (Manus, ChatGPT, Deepseek): Methodology Suggestions, Writing – Assistance in Organization and Drafting, Proofreading.

---

## Competing Interests

The authors declare no competing interests.

---

## Supplementary Materials

**Supplementary Documentation**: Complete mathematical derivations, computational implementation details, and parameter tuning guidelines are available in the accompanying technical documentation.

**Supplementary Section: Future Directions**: Detailed experimental validation designs are provided in a separate document outlining specific biological systems, perturbation strategies, and success criteria for validating scQDiff predictions.

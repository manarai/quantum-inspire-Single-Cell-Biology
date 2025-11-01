# Preface

The ability to understand and manipulate the trajectories that cells take as they differentiate, activate, or transform is one of the grand challenges of modern biology. Single-cell genomics has given us unprecedented snapshots of cellular states, yet the question remains: how do we move from these static portraits to a dynamic, mechanistic understanding of the regulatory forces that drive cellular decision-making?

This booklet introduces **scQDiff** (single-cell Quantum Diffusion), a conceptual and mathematical framework that addresses this challenge by drawing inspiration from an unexpected source: quantum mechanics. At its core, scQDiff leverages the Schrödinger Bridge problem—a mathematical formulation with deep roots in quantum physics—to learn continuous, time-varying gene regulatory networks from single-cell RNA sequencing data. By decomposing these networks into fundamental regulatory archetypes and quantifying the energy required to transition between cellular states, scQDiff provides both interpretability and predictive power.

## Who This Booklet Is For

This work is intended for a diverse audience spanning computational biology, biophysics, and applied mathematics. Graduate students and researchers in computational biology will find a rigorous yet accessible introduction to quantum-inspired methods for trajectory inference. Physicists and applied mathematicians entering the field of systems biology will appreciate the formal mathematical treatment and the explicit connections to familiar concepts from quantum mechanics and optimal control theory. Experimentalists seeking to design perturbation experiments or validate computational predictions will benefit from the detailed case studies and experimental validation designs.

We assume the reader has a basic familiarity with linear algebra, calculus, and probability theory. Some exposure to differential equations and optimization is helpful but not strictly required, as we build up the necessary concepts from first principles. No prior knowledge of quantum mechanics is assumed—Chapter 2 provides a self-contained introduction tailored specifically for biologists.

## Structure of the Booklet

The booklet is organized into four main parts, each building upon the previous one. **Part I: Foundations** establishes the biological motivation, introduces the quantum mechanics analogy, and lays out the mathematical groundwork of the Schrödinger Bridge problem. **Part II: Core Framework** presents the complete scQDiff pipeline, from learning drift fields to extracting regulatory archetypes and synthesizing guided trajectories. Each chapter includes detailed mathematical derivations, algorithmic pseudocode, and worked examples. **Part III: Extensions** explores how scQDiff can be generalized to incorporate multi-omic data, spatial information, and prior biological knowledge. **Part IV: Applications and Validation** demonstrates the framework's utility through detailed case studies and proposes concrete experimental designs for validation.

The appendices provide additional mathematical details, comparisons with existing methods, software resources, and a comprehensive glossary. Throughout the booklet, we emphasize both mathematical rigor and biological intuition, using tables, figures, and concrete examples to make abstract concepts tangible.

## A Note on Notation

We strive for consistency in notation throughout this work. Scalars are denoted by italic lowercase letters (*x*, *t*), vectors by bold lowercase letters (**x**), matrices by bold uppercase letters (**M**), and tensors by calligraphic uppercase letters (𝓙). Functions are written in roman type with their arguments (*f*(*x*, *t*)). The symbol ∇ denotes the gradient operator, and ∂/∂*x* denotes partial differentiation. A complete notation guide is provided at the beginning of Chapter 4.

## Acknowledgments

The development of scQDiff has been informed by conversations with researchers across multiple disciplines. We are grateful to the broader single-cell genomics community for creating the rich ecosystem of tools and datasets that make this work possible. Special thanks to the developers of SCENIC+, PINNACLE, and the various trajectory inference methods that have paved the way for this quantum-inspired approach.

## How to Use This Booklet

For readers primarily interested in the biological applications, we recommend starting with Chapter 1, skimming Chapters 2-3 for conceptual understanding, and then focusing on Chapters 7-9 and Part IV. For those seeking a complete mathematical understanding, a linear reading from Chapter 1 through the appendices is recommended. Computational practitioners implementing scQDiff should pay particular attention to Chapters 5-6 and Chapter 16, which provide implementation details and parameter selection guidelines.

We hope this booklet serves as both a comprehensive reference and an invitation to explore the deep connections between quantum physics and biology. The journey from electrons to cells is not merely an analogy—it is a testament to the universality of certain mathematical structures and the power of interdisciplinary thinking.

---

**Tommy W. Terooatea**  
**Manus AI**  
November 2025
# scQDiff: A Comprehensive Guide to Quantum-Inspired Single-Cell Trajectory Analysis

**Authors:** Tommy W. Terooatea, Manus AI

---

## Table of Contents

### Preface

### Part I: Foundations

**Chapter 1: Introduction to Single-Cell Trajectory Inference**
- 1.1 The Challenge of Cellular Decision-Making
- 1.2 From Snapshots to Dynamics
- 1.3 Limitations of Current Methods
- 1.4 The scQDiff Vision

**Chapter 2: Quantum Mechanics for Computational Biologists**
- 2.1 The Schrödinger Equation
- 2.2 Wavefunctions and Probability
- 2.3 Eigenstates and Energy Levels
- 2.4 The Quantum-to-Biology Analogy

**Chapter 3: The Schrödinger Bridge Problem**
- 3.1 From Brownian Motion to Optimal Transport
- 3.2 Entropic Regularization
- 3.3 The Schrödinger System
- 3.4 Connection to Quantum Mechanics

**Chapter 4: Mathematical Foundations of scQDiff**
- 4.1 Notation and Problem Setup
- 4.2 Stochastic Differential Equations
- 4.3 The Drift Field Concept
- 4.4 From Particles to Populations

### Part II: Core Framework

**Chapter 5: Learning the Drift Field**
- 5.1 Velocity-Constrained Field Learning
- 5.2 Bridge-Lite in Latent Space
- 5.3 Biological Constraints and Regularization
- 5.4 Implementation Considerations

**Chapter 6: Computing the Temporal Jacobian Tensor**
- 6.1 The Jacobian as Regulatory Influence
- 6.2 Local Ridge Regression Method
- 6.3 Autograd Jacobian Method
- 6.4 Tensor Assembly and Validation

**Chapter 7: Regulatory Archetype Extraction**
- 7.1 Tensor Decomposition via SVD
- 7.2 Biological Interpretation of Archetypes
- 7.3 Determining the Number of Archetypes
- 7.4 Validation Against Known Regulons

**Chapter 8: Guided Trajectory Synthesis**
- 8.1 Forward Prediction
- 8.2 Reverse Engineering
- 8.3 Archetype-Based Guidance
- 8.4 Uncertainty Quantification

**Chapter 9: Control Energy Theory and Applications**
- 9.1 Mathematical Derivation
- 9.2 Optimal Control Formulation
- 9.3 Connection to Archetypes
- 9.4 Cellular Reprogramming Example
- 9.5 Irreversibility and Round-Trip Error

### Part III: Extensions

**Chapter 10: Multi-Omic Integration**
- 10.1 Unified Multi-Modal Tensor
- 10.2 Cross-Modal Jacobian Blocks
- 10.3 scATAC-seq Integration
- 10.4 Protein and Metabolite Layers

**Chapter 11: Spatial Tensor Fields**
- 11.1 Space-Dependent Dynamics
- 11.2 Morphogen Gradients
- 11.3 Cell-Cell Communication
- 11.4 Spatial Niche Effects

**Chapter 12: Anchored Decomposition with Prior Networks**
- 12.1 Integration with SCENIC+
- 12.2 Integration with PINNACLE
- 12.3 Constrained Tensor Factorization
- 12.4 Benefits and Trade-offs

### Part IV: Applications and Validation

**Chapter 13: Case Study: T-Cell Activation**
- 13.1 Biological Background
- 13.2 Data and Preprocessing
- 13.3 scQDiff Analysis
- 13.4 Archetype Interpretation
- 13.5 Predictive Validation

**Chapter 14: Case Study: Cancer Progression and Reversal**
- 14.1 Modeling Oncogenic Transformation
- 14.2 Identifying Irreversible Transitions
- 14.3 Designing Combination Therapies
- 14.4 Control Energy Analysis

**Chapter 15: Experimental Validation Designs**
- 15.1 Validating the Temporal Jacobian
- 15.2 Validating Forward Trajectory Synthesis
- 15.3 Validating Reverse Reprogramming
- 15.4 Success Criteria and Metrics

**Chapter 16: Computational Implementation Guide**
- 16.1 Software Requirements
- 16.2 Data Preprocessing Pipeline
- 16.3 Parameter Selection
- 16.4 Scalability Considerations
- 16.5 Code Examples

### Appendices

**Appendix A: Mathematical Derivations**
- A.1 Schrödinger Bridge Derivation
- A.2 Jacobian Computation Details
- A.3 SVD and Low-Rank Approximation
- A.4 Control Energy Calculus of Variations

**Appendix B: Comparison with Existing Methods**
- B.1 RNA Velocity
- B.2 CellRank
- B.3 Dynamo
- B.4 SCENIC+
- B.5 Cflows

**Appendix C: Software and Data Resources**
- C.1 scQDiff Implementation
- C.2 Example Datasets
- C.3 Visualization Tools

**Appendix D: Glossary of Terms**

**References**

**Index**
# Chapter 1: Introduction to Single-Cell Trajectory Inference

The advent of single-cell RNA sequencing (scRNA-seq) has revolutionized our ability to profile the molecular state of individual cells within complex tissues. By measuring the expression levels of thousands of genes in tens or hundreds of thousands of cells simultaneously, scRNA-seq provides an unprecedented high-resolution view of cellular heterogeneity. Yet, for all its power, scRNA-seq presents us with a fundamental challenge: the data are inherently static snapshots, frozen moments in time, while the biological processes we seek to understand—differentiation, activation, disease progression—are fundamentally dynamic.

## 1.1 The Challenge of Cellular Decision-Making

Every cell in a developing embryo, an activated immune response, or a progressing tumor is constantly making decisions. These decisions are encoded in the cell's gene regulatory network (GRN), a complex web of interactions where transcription factors, signaling molecules, and epigenetic modifiers collectively determine which genes are turned on or off, and at what levels. Understanding how cells navigate this regulatory landscape is central to developmental biology, immunology, cancer research, and regenerative medicine.

Consider the differentiation of a hematopoietic stem cell into a mature blood cell. This process involves a cascade of regulatory decisions, each narrowing the cell's potential fates until it commits to a specific lineage—erythrocyte, lymphocyte, or myeloid cell. At each branch point, the cell's GRN must integrate external signals (cytokines, cell-cell contacts) with its internal state (chromatin accessibility, transcription factor levels) to determine the next step. Traditional bulk RNA-seq, which averages gene expression across millions of cells, obscures these individual trajectories. Single-cell methods reveal the diversity of paths cells can take, but they do not, by themselves, tell us the rules governing those paths.

The challenge, then, is to infer the dynamics—the forces, the regulatory logic, the decision-making processes—from static snapshots. This is akin to reconstructing the motion of a river from photographs of leaves floating on its surface at a single moment in time. We need a mathematical framework that can extract continuous trajectories, identify the underlying regulatory forces, and ultimately predict how cells will behave under new conditions or perturbations.

## 1.2 From Snapshots to Dynamics

The field of trajectory inference has made significant strides in addressing this challenge. Early methods, such as Monocle and Slingshot, focused on ordering cells along a pseudotime axis—a continuous variable that approximates the progression of cells through a biological process. These methods construct a low-dimensional embedding of the cell state space (typically using PCA or diffusion maps) and fit curves through this space to represent differentiation trajectories. While powerful for visualization and ordering, these approaches are largely phenomenological. They tell us *what* the trajectory looks like but provide limited insight into *why* cells follow that path.

More recent methods have sought to incorporate mechanistic information. RNA velocity, pioneered by La Manno and colleagues, uses the ratio of unspliced to spliced mRNA to infer the direction of gene expression changes, providing a velocity vector for each cell. This approach offers a glimpse into the immediate future of each cell's state, but it is limited to short timescales (hours) and does not directly model the regulatory interactions driving those changes. CellRank extends RNA velocity by combining it with Markov chain models to infer cell fate probabilities, while Dynamo learns a continuous vector field from velocity data to enable trajectory prediction.

Despite these advances, a critical gap remains: existing methods do not provide a unified framework for learning the time-varying structure of gene regulatory networks, decomposing them into interpretable components, and quantifying the difficulty of manipulating cellular trajectories. This is where scQDiff enters the picture.

## 1.3 Limitations of Current Methods

To motivate the development of scQDiff, it is instructive to consider the limitations of current trajectory inference methods in more detail. These limitations can be grouped into three broad categories: interpretability, temporal resolution, and predictive control.

**Interpretability** is a persistent challenge. Many trajectory inference methods produce smooth curves through cell state space, but the biological meaning of these curves—what regulatory programs are active at each point, which genes are influencing which others—is often opaque. Methods like SCENIC+ infer gene regulatory networks from scRNA-seq data, but they typically produce static networks that do not capture how regulatory interactions change over time. The result is a disconnect between the inferred trajectories and the underlying regulatory logic.

**Temporal resolution** is another issue. RNA velocity provides instantaneous snapshots of gene expression dynamics, but it does not model the continuous evolution of the regulatory network over longer timescales. Methods that do attempt to model continuous dynamics, such as Dynamo, often rely on strong assumptions about the form of the vector field (e.g., polynomial or neural network parameterizations) without grounding these assumptions in a principled probabilistic framework. This can lead to overfitting, poor generalization, and difficulty in quantifying uncertainty.

**Predictive control** is perhaps the most significant gap. If our ultimate goal is to engineer cellular behavior—to reprogram a diseased cell back to health, to direct a stem cell towards a desired fate, to block a cancer cell's progression—we need more than descriptive models. We need a framework that can quantify the difficulty of such interventions, identify the most efficient perturbations, and predict the outcomes of those perturbations with quantifiable uncertainty. Current methods offer limited support for this type of analysis.

## 1.4 The scQDiff Vision

scQDiff is designed to address these limitations by bringing together ideas from quantum mechanics, optimal transport theory, and control theory into a unified framework for single-cell trajectory analysis. The central innovation is the use of the **Schrödinger Bridge** problem as the mathematical foundation for learning cellular dynamics.

The Schrödinger Bridge, originally formulated by Erwin Schrödinger in 1931, asks: given two probability distributions at two different times, what is the most probable stochastic process that connects them? This problem has a deep connection to quantum mechanics—the solution involves potentials and wavefunctions analogous to those in the Schrödinger equation—but it is also a powerful tool for modeling any system where we observe distributions at discrete time points and wish to infer the continuous dynamics in between.

In the context of single-cell data, the Schrödinger Bridge allows us to learn a **drift field** *f*(*x*, *t*)—a time-varying vector field that describes the average velocity of a cell at state *x* and pseudotime *t*—from snapshots of cell distributions at different time points. Critically, this drift field is not arbitrary; it is the solution to an optimal transport problem, meaning it represents the most efficient, biologically plausible path for cells to evolve from one distribution to another.

From this drift field, scQDiff computes the **temporal Jacobian tensor**, a three-dimensional array that captures how the influence of each gene on every other gene changes over time. This tensor is then decomposed into a small number of **regulatory archetypes**—fundamental, time-independent patterns of gene-gene interactions—whose activations vary over time. These archetypes provide a compact, interpretable representation of the GRN's dynamics, analogous to the eigenstates of a quantum system.

Finally, scQDiff uses these archetypes to perform **guided trajectory synthesis**, predicting how cells will evolve under specific interventions, and to calculate the **control energy** required to steer a cell from one state to another. This energy provides a quantitative measure of the difficulty of cellular reprogramming, grounded in the learned regulatory architecture.

The vision of scQDiff, then, is to move beyond descriptive trajectory inference towards a predictive, mechanistic, and actionable understanding of cellular dynamics. By framing the problem through the lens of quantum-inspired mathematics, we gain not only new computational tools but also new biological intuition—a way of thinking about cells as quantum-like systems navigating potential landscapes, with archetypes as their fundamental modes and control energy as the cost of intervention.

In the chapters that follow, we will build this framework from the ground up, starting with the necessary mathematical foundations and culminating in detailed applications to real biological problems. The journey will take us from the Schrödinger equation to single-cell genomics, from abstract tensor decompositions to concrete reprogramming strategies. Along the way, we will see how ideas from physics can illuminate biology, and how biology, in turn, can inspire new mathematics.
# Chapter 2: Quantum Mechanics for Computational Biologists

For most biologists, quantum mechanics exists in a realm far removed from the messy, macroscopic world of cells and tissues. It is the domain of subatomic particles, exotic phenomena like superposition and entanglement, and equations filled with complex numbers and operators. Yet, as we will see in this chapter, the mathematical framework of quantum mechanics—particularly the Schrödinger equation—provides a surprisingly apt language for describing certain aspects of cellular dynamics. This chapter is not a comprehensive introduction to quantum physics; rather, it is a targeted primer designed to equip computational biologists with the specific concepts and intuitions needed to understand the quantum-inspired foundations of scQDiff.

## 2.1 The Schrödinger Equation

At the heart of non-relativistic quantum mechanics lies the **Schrödinger equation**, which governs the evolution of a quantum system's state over time. For a single particle (such as an electron) moving in a potential *V*(*x*, *t*), the time-dependent Schrödinger equation is:

$$i\hbar \frac{\partial \Psi(x,t)}{\partial t} = \hat{H} \Psi(x,t) = \left[ -\frac{\hbar^2}{2m} \nabla^2 + V(x,t) \right] \Psi(x,t)$$

Let us unpack this equation term by term, as each component has a biological analogue that will become important later.

The quantity Ψ(*x*, *t*) is the **wavefunction**, a complex-valued function that completely describes the state of the particle. The variable *x* represents the particle's position in space, and *t* is time. The wavefunction itself does not have a direct physical meaning, but its squared magnitude, |Ψ(*x*, *t*)|², gives the **probability density** of finding the particle at position *x* at time *t*. This probabilistic interpretation, known as the Born rule, is central to quantum mechanics and will have a direct parallel in scQDiff, where we work with probability distributions of cells in gene expression space.

On the right-hand side of the equation, we have the **Hamiltonian operator** *Ĥ*, which represents the total energy of the system. It consists of two parts: the kinetic energy term, -ħ²/(2*m*) ∇², and the potential energy term, *V*(*x*, *t*). The Laplacian operator ∇² = ∂²/∂*x*² (in one dimension) measures the curvature of the wavefunction, and when multiplied by -ħ²/(2*m*), it gives the kinetic energy contribution. The potential *V*(*x*, *t*) represents the external forces acting on the particle—for an electron in an atom, this is the attractive Coulomb potential from the nucleus.

The left-hand side of the equation, *i*ħ ∂Ψ/∂*t*, describes how the wavefunction changes over time. The presence of the imaginary unit *i* is what makes the Schrödinger equation fundamentally different from classical diffusion equations, leading to phenomena like interference and tunneling. The constant ħ (h-bar) is the reduced Planck's constant, ħ = *h*/(2π) ≈ 1.055 × 10⁻³⁴ J·s, which sets the scale of quantum effects.

The Schrödinger equation is a **linear partial differential equation**. This linearity has profound consequences: if Ψ₁ and Ψ₂ are solutions, then any linear combination *c*₁Ψ₁ + *c*₂Ψ₂ is also a solution. This is the principle of **superposition**, which allows a quantum particle to exist in a combination of multiple states simultaneously.

## 2.2 Wavefunctions and Probability

The wavefunction Ψ(*x*, *t*) encodes all the information about a quantum system, but extracting physical predictions from it requires careful interpretation. As mentioned, the Born rule states that the probability density of finding the particle at position *x* is:

$$P(x,t) = |\Psi(x,t)|^2 = \Psi^*(x,t) \Psi(x,t)$$

where Ψ* denotes the complex conjugate of Ψ. Since Ψ is generally complex, we must take its squared magnitude to obtain a real, non-negative probability density. The wavefunction must be normalized, meaning that the total probability of finding the particle somewhere in space is 1:

$$\int_{-\infty}^{\infty} |\Psi(x,t)|^2 \, dx = 1$$

This probabilistic nature of quantum mechanics is not due to ignorance or measurement error; it is a fundamental feature of the theory. Before measurement, the particle does not have a definite position—it exists in a superposition of all possible positions, weighted by the wavefunction. Only upon measurement does the wavefunction "collapse" to a specific outcome.

In the context of scQDiff, we will work with probability distributions *p*(*x*, *t*) that describe the distribution of cells in gene expression space *x* at pseudotime *t*. While these distributions are real-valued (not complex), the conceptual parallel is clear: just as the wavefunction describes the probabilistic state of a quantum particle, the cell distribution describes the probabilistic state of a population of cells undergoing a biological process.

## 2.3 Eigenstates and Energy Levels

When the potential *V*(*x*) is time-independent, we can seek special solutions to the Schrödinger equation of the form:

$$\Psi(x,t) = \psi(x) e^{-iEt/\hbar}$$

Substituting this into the time-dependent Schrödinger equation and canceling the time-dependent exponential factor, we obtain the **time-independent Schrödinger equation**:

$$\hat{H} \psi(x) = E \psi(x)$$

or, more explicitly:

$$-\frac{\hbar^2}{2m} \frac{d^2\psi}{dx^2} + V(x) \psi(x) = E \psi(x)$$

This is an **eigenvalue equation**. The functions *ψ*(*x*) that satisfy this equation are called **eigenstates** or **eigenfunctions**, and the corresponding values *E* are the **energy eigenvalues**. Each eigenstate represents a state of definite energy, and the set of all energy eigenvalues forms the **energy spectrum** of the system.

For many systems of physical interest—such as an electron in a hydrogen atom or a particle in a box—the energy spectrum is **discrete** (quantized). This quantization is a hallmark of quantum mechanics and gives rise to the term "quantum" itself. For example, the electron in a hydrogen atom can only occupy certain discrete energy levels, labeled by quantum numbers *n* = 1, 2, 3, .... Transitions between these levels involve the absorption or emission of photons with energy Δ*E* = *E<sub>m</sub>* - *E<sub>n</sub>* = *h*ν, where ν is the photon's frequency.

The eigenstates form a complete orthonormal basis, meaning any wavefunction Ψ(*x*, *t*) can be expressed as a linear combination (superposition) of eigenstates:

$$\Psi(x,t) = \sum_n c_n \psi_n(x) e^{-iE_n t/\hbar}$$

where the coefficients *c<sub>n</sub>* are complex numbers that determine the amplitude of each eigenstate in the superposition. The squared magnitude |*c<sub>n</sub>*|² gives the probability that a measurement of the system's energy will yield the value *E<sub>n</sub>*.

This decomposition into eigenstates is the quantum mechanical analogue of Fourier analysis or principal component analysis. Just as a complex signal can be decomposed into a sum of sine waves (Fourier modes) or a high-dimensional dataset can be decomposed into principal components, a quantum state can be decomposed into energy eigenstates. In scQDiff, we will perform an analogous decomposition of the temporal Jacobian tensor into **regulatory archetypes**, which serve as the fundamental "modes" of the gene regulatory network.

## 2.4 The Quantum-to-Biology Analogy

Having introduced the core concepts of quantum mechanics, we are now in a position to draw explicit parallels to the scQDiff framework. The table below summarizes the key correspondences:

| Quantum Mechanics | scQDiff Cellular Dynamics |
|:------------------|:--------------------------|
| **Wavefunction** Ψ(*x*, *t*) | **Cell Distribution** *p*(*x*, *t*) |
| Describes the state of a single particle | Describes the distribution of cells in gene space |
| **Probability Density** \|Ψ\|² | **Cell Density** *p*(*x*, *t*) |
| Probability of finding particle at *x* | Density of cells at gene expression state *x* |
| **Schrödinger Equation** *i*ħ ∂Ψ/∂*t* = *Ĥ*Ψ | **Schrödinger Bridge SDE** *dx* = *f*(*x*, *t*)*dt* + √(2ε)*dW* |
| Governs wavefunction evolution | Governs cell state evolution |
| **Potential** *V*(*x*) | **Drift Field** *f*(*x*, *t*) |
| External force field on particle | Regulatory force field on cell |
| **Hamiltonian** *Ĥ* = -ħ²/(2*m*)∇² + *V* | **Generator** of the SDE |
| Total energy operator | Infinitesimal generator of the stochastic process |
| **Energy Eigenstates** *ψ<sub>n</sub>* | **Regulatory Archetypes** *M<sub>k</sub>* |
| Fundamental modes of the quantum system | Fundamental modes of the gene regulatory network |
| **Energy Eigenvalues** *E<sub>n</sub>* | **Singular Values** *σ<sub>k</sub>* |
| Quantized energy levels | Importance/strength of each archetype |
| **Superposition** Ψ = Σ *c<sub>n</sub>ψ<sub>n</sub>* | **Archetype Decomposition** 𝓙(*t*) ≈ Σ *a<sub>k</sub>*(*t*)*M<sub>k</sub>* |
| State as a mix of eigenstates | GRN as a mix of archetypes |
| **Transition Energy** Δ*E* = ħω | **Control Energy** *E*<sub>ctrl</sub> = ∫ ‖*u*‖² *dt* |
| Energy to move between quantum states | Effort to reprogram cell state |

These parallels are not mere metaphors; they reflect deep structural similarities in the underlying mathematics. Both quantum mechanics and scQDiff deal with probabilistic evolution governed by differential equations, both involve decomposition into fundamental modes, and both provide a framework for quantifying the "cost" of transitions between states.

However, it is equally important to recognize the differences. The Schrödinger equation is fundamentally complex-valued and linear, leading to interference effects that have no direct biological analogue. The scQDiff framework, by contrast, works with real-valued gene expression vectors and learns dynamics from data rather than deriving them from first principles. The "potential" in scQDiff is not a fundamental force but an emergent property of the gene regulatory network. Despite these differences, the quantum mechanical framework provides invaluable intuition and mathematical tools that we will leverage throughout the rest of this booklet.

In the next chapter, we will introduce the Schrödinger Bridge problem, the mathematical bridge that connects quantum mechanics to optimal transport theory and, ultimately, to the inference of cellular trajectories from single-cell data.
# Chapter 3: The Schrödinger Bridge Problem

The Schrödinger Bridge problem, first formulated by Erwin Schrödinger in 1931, represents one of the most elegant connections between quantum mechanics, probability theory, and optimal transport. While Schrödinger originally posed the problem in the context of statistical mechanics and the foundations of quantum theory, it has since found applications in fields ranging from economics to machine learning. For scQDiff, the Schrödinger Bridge provides the mathematical foundation for inferring continuous cellular trajectories from discrete snapshots of cell distributions. This chapter introduces the problem, derives its solution, and establishes the connection to both quantum mechanics and single-cell trajectory inference.

## 3.1 From Brownian Motion to Optimal Transport

To understand the Schrödinger Bridge, we begin with the simpler problem of Brownian motion. Consider a particle undergoing random diffusion in space, described by the stochastic differential equation (SDE):

$$dx_t = \sqrt{2\varepsilon} \, dW_t$$

where *dW<sub>t</sub>* is a Wiener process (standard Brownian motion) and ε is the diffusion coefficient. This SDE has no drift term—the particle's motion is purely random. If the particle starts at position *x*₀ at time *t* = 0, its position at time *t* = 1 is normally distributed with mean *x*₀ and variance 2ε.

Now suppose we observe not a single particle but an entire ensemble (population) of particles. Let *p*₀(*x*) be the initial distribution of particles at *t* = 0, and let *p*₁(*x*) be the distribution at *t* = 1. If the particles evolve according to pure Brownian motion (no drift), the distribution *p<sub>t</sub>*(*x*) at any intermediate time *t* ∈ [0, 1] is determined by the heat equation (Fokker-Planck equation):

$$\frac{\partial p_t}{\partial t} = \varepsilon \nabla^2 p_t$$

This equation describes the diffusion of probability density over time. Given *p*₀, we can solve this equation forward in time to obtain *p*₁. However, in general, the resulting *p*₁ will not match an arbitrary target distribution *p*₁<sup>target</sup> that we might specify.

This brings us to the central question of the Schrödinger Bridge problem: **Given two distributions *p*₀ and *p*₁ at times *t* = 0 and *t* = 1, what is the most probable stochastic process (diffusion) that starts at *p*₀ and ends at *p*₁?**

The phrase "most probable" requires clarification. Among all possible diffusion processes that connect *p*₀ and *p*₁, we seek the one that is "closest" to the reference Brownian motion. This closeness is measured using the **Kullback-Leibler (KL) divergence** between the path distributions. Intuitively, we want to find the minimal perturbation to Brownian motion—the minimal drift—that achieves the desired endpoint distribution.

## 3.2 Entropic Regularization and the Schrödinger System

The Schrödinger Bridge problem can be formulated as an **entropic optimal transport** problem. Classical optimal transport, as formulated by Monge and Kantorovich, seeks to find the most efficient way to transport mass from one distribution to another, minimizing a cost function (typically the squared distance). The Schrödinger Bridge adds a crucial ingredient: **entropic regularization**, which penalizes transport plans that deviate too far from the natural diffusion process.

Mathematically, the Schrödinger Bridge problem is:

$$\min_{\pi \in \Pi(p_0, p_1)} \text{KL}(\pi \| \pi_{\text{ref}})$$

where:
- π is a joint distribution (coupling) over paths that has marginals *p*₀ at *t* = 0 and *p*₁ at *t* = 1.
- Π(*p*₀, *p*₁) is the set of all such couplings.
- π<sub>ref</sub> is the reference measure, corresponding to pure Brownian motion.
- KL(π ‖ π<sub>ref</sub>) is the Kullback-Leibler divergence, measuring the "distance" between π and π<sub>ref</sub>.

The solution to this problem is given by the **Schrödinger system**, a pair of coupled equations for two potential functions *φ*(*x*, *t*) and *ψ*(*x*, *t*):

$$p_t(x) = e^{\phi(x,t)} \cdot \left( e^{-\varepsilon \nabla^2 (1-t)} e^{\psi(\cdot, 1)} \right)(x) \cdot e^{-\varepsilon \nabla^2 t} e^{\phi(\cdot, 0)}(x)$$

This equation, while elegant, is quite abstract. A more intuitive formulation involves the **forward and backward Schrödinger potentials**, which define a modified drift field. The key result is that the optimal diffusion process has the form:

$$dx_t = f(x_t, t) \, dt + \sqrt{2\varepsilon} \, dW_t$$

where the drift *f*(*x*, *t*) is given by:

$$f(x, t) = \varepsilon \nabla \log p_t(x) + \varepsilon \nabla \phi(x, t)$$

The first term, ε ∇ log *p<sub>t</sub>*, is the **score function** of the distribution *p<sub>t</sub>*. It represents the direction in which the probability density is increasing most rapidly. The second term, ε ∇ *φ*, is the gradient of the Schrödinger potential, analogous to a force field.

The Schrödinger Bridge solution has a remarkable property: it is the **unique** diffusion process that minimizes the KL divergence to the reference Brownian motion while satisfying the marginal constraints *p*₀ and *p*₁. This uniqueness is what makes the Schrödinger Bridge a principled choice for trajectory inference.

## 3.3 The Schrödinger System: Forward and Backward Equations

To gain further insight, we can express the Schrödinger Bridge solution in terms of forward and backward processes. Define:
- *φ*(*x*, *t*): the **forward potential**, which encodes information from the initial distribution *p*₀.
- *ψ*(*x*, *t*): the **backward potential**, which encodes information from the final distribution *p*₁.

These potentials satisfy a system of partial differential equations known as the **Schrödinger system**:

$$\begin{cases}
\frac{\partial \phi}{\partial t} = \varepsilon \nabla^2 \phi + \frac{1}{2} |\nabla \phi|^2 \\
\frac{\partial \psi}{\partial t} = -\varepsilon \nabla^2 \psi - \frac{1}{2} |\nabla \psi|^2
\end{cases}$$

with boundary conditions *φ*(*x*, 0) = log *p*₀(*x*) and *ψ*(*x*, 1) = log *p*₁(*x*).

These equations are nonlinear and coupled through the marginal distribution *p<sub>t</sub>*(*x*), which is given by:

$$p_t(x) \propto e^{\phi(x,t) + \psi(x,t)}$$

The Schrödinger system can be solved iteratively using the **Sinkhorn algorithm**, which alternates between updating *φ* and *ψ* until convergence. This algorithm is computationally efficient and forms the basis for many modern implementations of entropic optimal transport.

Once the potentials *φ* and *ψ* are known, the drift field *f*(*x*, *t*) can be computed as:

$$f(x, t) = \varepsilon \nabla (\phi(x,t) + \psi(x,t))$$

This drift field is precisely what we need for scQDiff: it describes the average velocity of a cell at state *x* and pseudotime *t*, learned from the observed distributions *p*₀ and *p*₁.

## 3.4 Connection to Quantum Mechanics

The connection between the Schrödinger Bridge and quantum mechanics is more than nominal. Consider the time-independent Schrödinger equation for a particle in a potential *V*(*x*):

$$-\frac{\hbar^2}{2m} \nabla^2 \psi + V(x) \psi = E \psi$$

If we perform a transformation *ψ*(*x*) = *e*<sup>*φ*(*x*)</sup>, this equation becomes:

$$-\frac{\hbar^2}{2m} \left( \nabla^2 \phi + |\nabla \phi|^2 \right) + V(x) = E$$

This is structurally identical to the forward Schrödinger potential equation (with *t* held fixed). The term ∇²*φ* + |∇*φ*|² is known as the **Bohm potential** in the de Broglie-Bohm (pilot wave) interpretation of quantum mechanics. In this interpretation, the wavefunction *ψ* guides the motion of particles via a quantum potential, much like how the Schrödinger potentials *φ* and *ψ* guide the diffusion of particles in the Schrödinger Bridge.

This deep connection suggests that the Schrödinger Bridge is not merely a useful mathematical tool but a fundamental framework for describing probabilistic dynamics. Just as the Schrödinger equation governs the evolution of quantum systems, the Schrödinger Bridge governs the evolution of probability distributions under optimal transport constraints.

In the context of scQDiff, this connection provides both mathematical rigor and physical intuition. The drift field *f*(*x*, *t*) learned by scQDiff is not an arbitrary vector field but the solution to a well-posed optimization problem with roots in quantum mechanics. The regulatory archetypes we extract from the Jacobian tensor are analogous to energy eigenstates, and the control energy we calculate for reprogramming is analogous to transition energies in quantum systems.

## 3.5 From Theory to Practice: The Schrödinger Bridge in scQDiff

In practice, applying the Schrödinger Bridge to single-cell data involves several steps:

1. **Data Preprocessing**: Single-cell RNA-seq data are preprocessed to obtain cell state vectors *x* ∈ ℝ<sup>G</sup> (typically after dimensionality reduction or feature selection). Cells are assigned to pseudotime bins, yielding empirical distributions *p*₀, *p*₁, ..., *p<sub>T</sub>* at discrete time points.

2. **Pairwise Bridge Fitting**: For each consecutive pair of distributions (*p<sub>t</sub>*, *p*<sub>t+1</sub>), we solve the Schrödinger Bridge problem to obtain the drift field *f*(*x*, *t*) that optimally transports *p<sub>t</sub>* to *p*<sub>t+1</sub>. This can be done using the Sinkhorn algorithm or, for high-dimensional data, using neural network parameterizations of the potentials.

3. **Drift Field Assembly**: The pairwise drift fields are stitched together to form a continuous drift field *f*(*x*, *t*) over the entire pseudotime range [0, 1].

4. **Jacobian Computation**: The Jacobian ∂*f*/∂*x* is computed at each time point, yielding the temporal Jacobian tensor 𝓙(*t*).

5. **Archetype Extraction**: The tensor 𝓙(*t*) is decomposed via SVD to obtain regulatory archetypes *M<sub>k</sub>* and their temporal activations *a<sub>k</sub>*(*t*).

This pipeline, which we will detail in Part II of this booklet, transforms the abstract mathematics of the Schrödinger Bridge into a concrete computational framework for single-cell trajectory analysis. The key insight is that the Schrödinger Bridge provides a principled way to infer continuous dynamics from discrete snapshots, grounded in both optimal transport theory and quantum mechanics.

In the next chapter, we will formalize the mathematical setup of scQDiff, introducing the notation, assumptions, and problem statement that will guide the rest of the booklet.
'''# Chapter 4: Mathematical Foundations of scQDiff

With the conceptual groundwork laid in the previous chapters, we now turn to the formal mathematical foundations of the scQDiff framework. This chapter will introduce the notation used throughout the booklet, define the stochastic differential equation (SDE) framework that models individual cell dynamics, connect these microscopic dynamics to the macroscopic evolution of cell populations via the Fokker-Planck equation, and culminate in a precise statement of the problem that scQDiff aims to solve.

## 4.1 Notation and Problem Setup

Consistent and clear notation is essential for navigating the mathematical landscape of scQDiff. The table below defines the key symbols and their meanings. We will adhere to this notation for the remainder of the booklet.

| Symbol | Description | Dimensions |
| :--- | :--- | :--- |
| *t* | Pseudotime, normalized to the interval [0, 1]. | Scalar |
| *G* | Number of genes or features in the cell state. | Scalar |
| **x** | Cell state vector, representing gene expression levels. | ℝ<sup>G</sup> |
| *p*(*x*, *t*) | Probability density of cells at state **x** and time *t*. | Scalar-valued function |
| *f*(*x*, *t*) | Drift field, the average velocity of a cell at state **x** and time *t*. | ℝ<sup>G</sup>-valued function |
| *ε* | Diffusion coefficient, representing biological stochasticity. | Scalar |
| *W<sub>t</sub>* | A standard G-dimensional Wiener process (Brownian motion). | ℝ<sup>G</sup>-valued process |
| 𝓙(*t*) | The Jacobian matrix of the drift field at time *t*. | ℝ<sup>G×G</sup> |
| 𝓙 | The temporal Jacobian tensor, stacked across time. | ℝ<sup>G×G×T</sup> |
| *M<sub>k</sub>* | The *k*-th regulatory archetype matrix. | ℝ<sup>G×G</sup> |
| *a<sub>k</sub>*(*t*) | The temporal activation profile of the *k*-th archetype. | Scalar-valued function |
| *u*(*x*, *t*) | External control input for guided synthesis. | ℝ<sup>G</sup>-valued function |
| *E*<sub>ctrl</sub> | Control energy required for a state transition. | Scalar |

The fundamental data for our problem are single-cell measurements. We assume we have a dataset of *N* cells, {**x**<sub>1</sub>, **x**<sub>2</sub>, ..., **x**<sub>N</sub>}, where each **x**<sub>i</sub> is a vector in ℝ<sup>G</sup>. Furthermore, we assume these cells have been assigned a pseudotime value *t<sub>i</sub>* ∈ [0, 1]. These data are treated as samples from an underlying, continuous probability distribution *p*(*x*, *t*).

## 4.2 The Stochastic Differential Equation (SDE) Framework

scQDiff models the trajectory of an individual cell as a path evolving according to a **stochastic differential equation (SDE)**. An SDE is a differential equation in which one or more of the terms is a stochastic process, providing a way to model systems that are subject to random fluctuations. The general form of an SDE is:

$$d\mathbf{x}_t = f(\mathbf{x}_t, t) \, dt + g(\mathbf{x}_t, t) \, d\mathbf{W}_t$$

In the scQDiff framework, we simplify this to the following form:

$$d\mathbf{x}_t = f(\mathbf{x}_t, t) \, dt + \sqrt{2\varepsilon} \, d\mathbf{W}_t \quad (*)$$

Let's break down this core equation:

- **d**x**<sub>t</sub>**: This represents an infinitesimal change in the cell's state vector **x** over an infinitesimal time interval *dt*.
- *f*(*x<sub>t</sub>*, *t*) **dt**: This is the **drift term**. The function *f*(*x*, *t*) is a vector field that specifies the deterministic part of the dynamics—the average velocity of a cell at state **x** and time *t*. It represents the collective effect of the gene regulatory network pushing the cell in a particular direction in gene expression space.
- **√2ε d**W**<sub>t</sub>**: This is the **diffusion term**. It models the stochastic or random component of the cell's motion. *W<sub>t</sub>* is a Wiener process, the mathematical formalization of Brownian motion. Intuitively, at each infinitesimal time step, this term adds a small, random "kick" to the cell's state, drawn from a Gaussian distribution. The scalar *ε* is the **diffusion coefficient**, which controls the magnitude of these random fluctuations. It represents the intrinsic biological noise in gene expression, transcriptional bursting, and other stochastic cellular events.

Equation (*) provides a complete model for the motion of a single cell. The cell attempts to follow the deterministic path laid out by the drift field *f*, but it is constantly perturbed by the random noise of the diffusion term. The resulting trajectory is a "drunken walk" along the flow lines of the vector field.

## 4.3 The Fokker-Planck Equation: From Particles to Populations

The SDE in the previous section describes the behavior of a single cell. However, our data consist of a population of cells. How do we connect the microscopic dynamics of individual cells to the macroscopic evolution of the entire cell population? The answer lies in the **Fokker-Planck equation**.

The Fokker-Planck equation is a partial differential equation (PDE) that describes the time evolution of the probability density function *p*(*x*, *t*) of a stochastic process. For the SDE given by Equation (*), the corresponding Fokker-Planck equation is:

$$\frac{\partial p(\mathbf{x},t)}{\partial t} = -\nabla \cdot [f(\mathbf{x},t) p(\mathbf{x},t)] + \varepsilon \nabla^2 p(\mathbf{x},t)$$

Where:
- ∇⋅ is the divergence operator.
- ∇² is the Laplacian operator.

This equation has two main terms on the right-hand side:

1.  **Advection Term**: -∇⋅[*f*(*x*,*t*) *p*(*x*,*t*)] describes how the probability density is carried along by the drift field *f*. It is analogous to the continuity equation in fluid dynamics, stating that the change in density is due to the flow of probability into or out of a region.
2.  **Diffusion Term**: *ε*∇²*p*(*x*,*t*) describes how the probability density spreads out over time due to the random fluctuations. It is identical in form to the heat equation.

The Fokker-Planck equation provides the crucial link between the microscopic (SDE) and macroscopic (PDE) descriptions. If we know the drift field *f*, we can simulate the trajectories of thousands of individual cells using the SDE. Alternatively, we can solve the Fokker-Planck equation to see how the entire probability distribution of cells evolves over time. In scQDiff, we work in the other direction: we observe the population distributions *p*(*x*, *t*) at discrete time points and use the Schrödinger Bridge framework to infer the underlying drift field *f*(*x*, *t*).

## 4.4 The scQDiff Problem Statement

We are now ready to state the central problem of the scQDiff framework formally.

**Given:**

A dataset of *N* single-cell measurements {**x**<sub>i</sub>}<sub>i=1</sub><sup>N</sup>, where each **x**<sub>i</sub> ∈ ℝ<sup>G</sup>. These cells are sampled from an unknown, continuous biological process. Through pseudotime analysis, we obtain empirical approximations of the cell state distributions, *p̂*(*x*, *t<sub>j</sub>*), at a series of discrete time points *t*<sub>0</sub>, *t*<sub>1</sub>, ..., *t<sub>T</sub>* spanning the interval [0, 1].

**Goal:**

1.  **Infer the Dynamics**: Find the continuous, time-varying drift field *f*(*x*, *t*) of an SDE (*dx<sub>t</sub>* = *f*(*x<sub>t</sub>*, *t*)*dt* + √2ε*dW<sub>t</sub>*) such that the evolution of its probability density *p*(*x*, *t*) according to the Fokker-Planck equation is consistent with the observed distributions *p̂*(*x*, *t<sub>j</sub>*). This inference is performed by solving the Schrödinger Bridge problem between consecutive time points, which finds the drift field that is "closest" to pure Brownian motion while satisfying the endpoint distributional constraints.

2.  **Extract Regulatory Logic**: From the inferred drift field *f*(*x*, *t*), compute the **temporal Jacobian tensor** 𝓙(*t*) = ∇*f*(*x*, *t*). Decompose this tensor into a small number of interpretable **regulatory archetypes** *M<sub>k</sub>* and their **temporal activations** *a<sub>k</sub>*(*t*), such that 𝓙(*t*) ≈ Σ *a<sub>k</sub>*(*t*)*M<sub>k</sub>*.

3.  **Perform Predictive Control**: Use the learned drift field and archetypes to:
    a.  **Synthesize guided trajectories**: Predict the evolution of cells under external control inputs *u*(*x*, *t*).
    b.  **Calculate control energy**: Quantify the difficulty (*E*<sub>ctrl</sub>) of steering a cell from an initial state to a target state.

This problem statement encapsulates the entire scQDiff vision. It moves from data (snapshots of cell distributions) to dynamics (the drift field), from dynamics to mechanism (the regulatory archetypes), and from mechanism to prediction and control (guided synthesis and control energy). The following parts of this booklet will detail the solution to each of these goals, step by step.
''
'''# Chapter 5: Learning the Drift Field

The drift field, *f*(*x*, *t*), is the heart of the scQDiff framework. It is the time-varying vector field that dictates the deterministic component of a cell's motion through gene expression space. Inferring this high-dimensional function from sparse, noisy single-cell data is the first and most critical computational challenge. This chapter details two practical, complementary approaches for learning the drift field: **Velocity-Constrained Field Learning**, which leverages RNA velocity for high temporal resolution, and **Bridge-Lite in Latent Space**, which uses the Schrödinger Bridge framework for robust inference between discrete, distant time points.

## 5.1 The Challenge of High-Dimensional Inference

Directly learning the drift field *f*(*x*, *t*) in the full *G*-dimensional gene space (where *G* can be >20,000) is a formidable task. The data are sparse (many zero counts), noisy, and typically only available at a few discrete time points. A naive approach, such as fitting a neural network to predict the change in gene expression, would be highly prone to overfitting and would struggle to connect dynamics across large time gaps.

To make the problem tractable, scQDiff employs a combination of dimensionality reduction, principled probabilistic modeling, and the incorporation of biological prior knowledge in the form of RNA velocity. The two methods presented below can be used independently or in combination, depending on the nature of the available data.

## 5.2 Method 1: Velocity-Constrained Field Learning

This method is best suited for datasets where RNA velocity information is available and reliable. RNA velocity, derived from the ratio of unspliced to spliced mRNA transcripts, provides an estimate of the instantaneous rate of change of gene expression for each cell. It gives us a noisy, high-dimensional vector, **v**<sub>i</sub>, for each cell *i* that points in the direction of its future state.

Instead of directly learning the drift field, we learn a function that *predicts* the RNA velocity vectors. This approach reframes the problem from one of density estimation (as in the Schrödinger Bridge) to one of vector field regression, which is often more stable and data-efficient.

### 5.2.1 The Objective Function

We parameterize the drift field *f*(*x*, *t*) using a neural network, *f*<sub>θ</sub>(*x*, *t*), where θ are the network weights. The objective is to find the parameters θ that minimize the cosine dissimilarity between the predicted drift and the observed RNA velocity vectors across all cells:

$$\min_{\theta} \sum_{i=1}^{N} \left( 1 - \frac{f_{\theta}(\mathbf{x}_i, t_i) \cdot \mathbf{v}_i}{\|f_{\theta}(\mathbf{x}_i, t_i)\|_2 \|\mathbf{v}_i\|_2} \right)$$

This objective function encourages the learned drift field to align with the directions indicated by RNA velocity. The use of cosine similarity, rather than mean squared error, makes the method robust to the unknown scaling factor of RNA velocity vectors.

### 5.2.2 Incorporating Pseudotime

The drift field is a function of both cell state **x** and pseudotime *t*. The neural network *f*<sub>θ</sub> takes both as input. The pseudotime *t* can be derived from a variety of methods, such as diffusion pseudotime (DPT) or by fitting a principal curve through the data. By including *t* as an input, the network learns a smooth, time-varying vector field rather than a static one.

### 5.2.3 Advantages and Limitations

**Advantages:**
-   **High Temporal Resolution**: Leverages the instantaneous nature of RNA velocity to capture fine-grained dynamics.
-   **Data Efficiency**: Vector field regression is often more stable than density-based methods, especially in high dimensions.
-   **Directly Models Mechanism**: The learned field directly represents the forces driving gene expression changes.

**Limitations:**
-   **Requires RNA Velocity**: This method is only applicable to datasets with reliable splicing information (e.g., from 10x Genomics 3' or 5' protocols).
-   **Short-Term Dynamics**: RNA velocity is most reliable for short-term predictions (hours). It may not accurately capture long-range dynamics over days or weeks.
-   **Velocity Estimation Errors**: The method is sensitive to errors in the estimation of RNA velocity, which can be substantial for some genes or cell types.

## 5.3 Method 2: Bridge-Lite in Latent Space

This method is designed for datasets where RNA velocity is unavailable or unreliable, or for processes that occur over long timescales where velocity information is less relevant. It is a practical adaptation of the full Schrödinger Bridge problem, making it computationally feasible for large single-cell datasets.

### 5.3.1 The "Lite" Approach: Latent Space Optimal Transport

Instead of solving the Schrödinger Bridge problem in the full *G*-dimensional gene space, we first project the data into a low-dimensional latent space (typically 10-50 dimensions) using a method like PCA or a variational autoencoder (VAE). Let *z* = encoder(**x**) be the latent representation of a cell state **x**.

We then solve the Schrödinger Bridge problem in this low-dimensional latent space. We are given empirical distributions of latent cell states, *p̂*(*z*, *t<sub>j</sub>*), at discrete time points *t*<sub>0</sub>, *t*<sub>1</sub>, ..., *t<sub>T</sub>*. For each consecutive pair of time points (*t<sub>j</sub>*, *t*<sub>j+1</sub>*), we use the Sinkhorn algorithm to solve the entropic optimal transport problem between *p̂*(*z*, *t<sub>j</sub>*) and *p̂*(*z*, *t*<sub>j+1</sub>*). This yields an optimal transport plan, **P**<sub>j</sub>, which can be thought of as a probabilistic mapping from cells at time *t<sub>j</sub>* to cells at time *t*<sub>j+1</sub>*.

### 5.3.2 Inferring the Drift

From the optimal transport plan **P**<sub>j</sub>, we can compute a displacement vector for each cell *i* at time *t<sub>j</sub>*:

$$\Delta \mathbf{z}_i = \sum_{k} \mathbf{P}_{j,ik} (\mathbf{z}_k - \mathbf{z}_i)$$

where the sum is over all cells *k* at time *t*<sub>j+1</sub>*. This displacement vector represents the average direction a cell at latent state **z**<sub>i</sub> moves to get to the next time point. The latent drift field, *f<sub>z</sub>*(*z*, *t*), can then be approximated by these displacement vectors, scaled by the time interval: *f<sub>z</sub>*(*z<sub>i</sub>*, *t<sub>j</sub>*) ≈ Δ**z**<sub>i</sub> / (*t*<sub>j+1</sub> - *t<sub>j</sub>*).

Finally, we fit a neural network *f<sub>z,θ</sub>*(*z*, *t*) to these latent displacement vectors. To get the drift field in the original gene space, we can use the decoder of the VAE or an appropriate inverse mapping.

### 5.3.3 Advantages and Limitations

**Advantages:**
-   **No Velocity Required**: Applicable to a wide range of datasets, including those from protocols that do not capture splicing information.
-   **Long-Range Dynamics**: By directly connecting distributions at distant time points, this method is well-suited for modeling long-term processes like development.
-   **Probabilistically Principled**: Grounded in the mathematics of optimal transport and the Schrödinger Bridge, providing a robust and well-posed formulation.

**Limitations:**
-   **Depends on Latent Space**: The quality of the inferred drift depends heavily on the quality of the low-dimensional embedding. Important biological variation may be lost during dimensionality reduction.
-   **Lower Temporal Resolution**: The dynamics are inferred between discrete time points, which may miss rapid, transient changes that occur between samples.
-   **Computational Cost**: While more efficient than a full-dimensional bridge, the Sinkhorn algorithm can still be computationally intensive for very large numbers of cells.

## 5.4 Combining the Methods and Implementation Considerations

The two methods are not mutually exclusive. For datasets with RNA velocity, one can use the velocity-constrained method to learn a high-resolution drift field and use the Bridge-Lite method to provide long-range constraints, ensuring that the short-term dynamics are consistent with the overall population-level shifts. This can be achieved by adding a term to the velocity-matching objective function that penalizes deviations from the Bridge-Lite-inferred drift.

In practice, the drift field *f*<sub>θ</sub>(*x*, *t*) is implemented as a multi-layer perceptron (MLP) with residual connections. The inputs are the cell state vector **x** (or its latent representation **z**) and the scalar pseudotime *t*. The network outputs a vector of the same dimension as the input state, representing the drift.

**Parameter Selection**: The key parameters to tune are the architecture of the neural network (number of layers, hidden units), the learning rate, the number of training epochs, and, for the Bridge-Lite method, the entropic regularization parameter ε in the Sinkhorn algorithm. A higher ε results in a more diffuse, less deterministic transport plan, while a lower ε results in a more focused, deterministic plan.

With a learned drift field *f*(*x*, *t*) in hand, we are now ready to probe the structure of the underlying regulatory network. The next chapter will detail how we compute the temporal Jacobian tensor, the object that quantifies the dynamic gene-gene influences that give rise to the drift field.
'''
'''# Chapter 6: Computing the Temporal Jacobian Tensor

The drift field, *f*(*x*, *t*), learned in the previous chapter, provides a macroscopic view of cellular dynamics, describing the average velocity of cells as they traverse the gene expression landscape. To move from this macroscopic description to a microscopic, mechanistic understanding of the gene regulatory network (GRN), we must analyze the local structure of this field. The mathematical tool for this is the **Jacobian matrix**, which captures how a small change in one input variable affects all the output variables of a vector function. By computing the Jacobian of the drift field, we can quantify the influence of each gene on the expression rate of every other gene, and by doing so over time, we construct the **temporal Jacobian tensor**—the central object for interpreting the GRN dynamics in scQDiff.

## 6.1 The Jacobian as Regulatory Influence

Recall that the drift field *f*(*x*, *t*) is a vector-valued function that takes a cell state **x** ∈ ℝ<sup>G</sup> and a time *t* and returns a velocity vector in ℝ<sup>G</sup>. The Jacobian matrix of the drift field with respect to the cell state **x** is a *G* × *G* matrix, denoted 𝓙(*t*), where each entry is a partial derivative:

$$J_{ij}(t) = \frac{\partial f_i(\mathbf{x},t)}{\partial x_j}$$

Let's unpack the biological meaning of this entry. *f<sub>i</sub>* is the *i*-th component of the drift vector, representing the rate of change of gene *i*. *x<sub>j</sub>* is the expression level of gene *j*. Therefore, the partial derivative *J<sub>ij</sub>*(*t*) quantifies the instantaneous effect of a small change in the expression of gene *j* on the rate of change of gene *i* at a specific point in time and state space. In other words:

-   If *J<sub>ij</sub>*(*t*) > 0, gene *j* **activates** gene *i*.
-   If *J<sub>ij</sub>*(*t*) < 0, gene *j* **represses** gene *i*.
-   If *J<sub>ij</sub>*(*t*) ≈ 0, gene *j* has no direct, instantaneous influence on gene *i*.

The entire matrix 𝓙(*t*) can thus be interpreted as the **adjacency matrix of the effective gene regulatory network** at time *t*. The diagonal entries, *J<sub>ii</sub>*(*t*), represent self-regulation—how the expression level of gene *i* affects its own rate of change (e.g., negative self-regulation or degradation).

By stacking these Jacobian matrices across all time points *t* ∈ [0, 1], we form the **temporal Jacobian tensor**, 𝓙 ∈ ℝ<sup>G×G×T</sup>. This three-dimensional object is the complete, time-varying representation of the GRN dynamics as learned by scQDiff.

## 6.2 Method 1: Local Ridge Regression

If the drift field was learned using the Bridge-Lite method (Chapter 5.3), we do not have an explicit analytical form for *f*(*x*, *t*). Instead, we have a set of displacement vectors Δ**z** in a latent space. In this case, we can estimate the Jacobian using a local regression approach.

For each cell *i* with latent state **z**<sub>i</sub> and displacement vector Δ**z**<sub>i</sub>, we consider its local neighborhood of *k* nearest neighbors. Within this neighborhood, we assume the drift field is approximately linear. We can then fit a linear model to predict the displacement vectors of the neighbors from their latent states:

$$\Delta \mathbf{z}_j \approx \mathbf{J}_z(\mathbf{z}_i) (\mathbf{z}_j - \mathbf{z}_i) + \text{offset}$$

where the sum is over the *k* neighbors *j* of cell *i*. This is a multivariate linear regression problem, where the goal is to find the latent Jacobian matrix **J**<sub>z</sub>(**z**<sub>i</sub>) that best explains the local flow. To prevent overfitting, especially in sparse regions of the data, we use **ridge regression**, which adds an L2 penalty to the magnitude of the Jacobian entries.

Once the latent Jacobian **J**<sub>z</sub> is estimated, it must be projected back into the full gene space. If a linear dimensionality reduction method like PCA was used, this is a straightforward linear transformation. If a non-linear method like a VAE was used, it involves multiplying by the Jacobian of the decoder network.

**Advantages:**
-   Does not require an analytical form for the drift field.
-   Robust to noise by averaging over local neighborhoods.

**Limitations:**
-   Computationally intensive, as it requires a separate regression for each cell or a representative set of cells.
-   The choice of neighborhood size *k* is a critical parameter.

## 6.3 Method 2: Autograd Jacobian

If the drift field was learned using the velocity-constrained method (Chapter 5.2), we have an explicit neural network parameterization, *f*<sub>θ</sub>(*x*, *t*). In this case, computing the Jacobian is much more direct. Modern deep learning frameworks like PyTorch and TensorFlow have built-in **automatic differentiation** (autograd) capabilities.

Given the trained network *f*<sub>θ</sub>, we can directly compute the Jacobian matrix ∇*f*<sub>θ</sub>(*x*, *t*) for any state **x** and time *t*. This is done by backpropagating the gradients of each output component of the network with respect to each input component. The process is highly efficient and exact (up to numerical precision).

### Pseudocode for Autograd Jacobian Computation

1.  **Select evaluation points**: Choose a set of representative cell states {**x**<sub>i</sub>} and time points {*t<sub>j</sub>*} at which to compute the Jacobian.
2.  **Initialize tensor**: Create an empty tensor 𝓙 of size *G* × *G* × *T*.
3.  **Loop over time points** *t<sub>j</sub>*:
    a.  Select a representative cell state **x**<sub>avg</sub> for this time point (e.g., the centroid of cells in that time bin).
    b.  Use the autograd engine to compute the Jacobian matrix **J** = ∇*f*<sub>θ</sub>(**x**<sub>avg</sub>, *t<sub>j</sub>*).
    c.  Store **J** in the corresponding slice of the tensor: 𝓙[:, :, *j*] = **J**.
4.  **Return** the assembled temporal Jacobian tensor 𝓙.

**Advantages:**
-   **Computationally Efficient**: Leverages the highly optimized backpropagation algorithms in deep learning libraries.
-   **Exact**: Provides the exact Jacobian of the learned neural network model.

**Limitations:**
-   **Requires a Differentiable Model**: Only applicable if the drift field is represented by a differentiable function, such as a neural network.
-   **Model-Dependent**: The resulting Jacobian is specific to the learned *f*<sub>θ</sub>. If the network has learned spurious correlations, the Jacobian will reflect them.

## 6.4 Tensor Assembly and Validation

Regardless of the method used, the result is the temporal Jacobian tensor, 𝓙 ∈ ℝ<sup>G×G×T</sup>. This is a rich, high-dimensional object that contains the full time-varying regulatory information learned by the model.

Before proceeding to decomposition, it is often useful to perform some validation checks on the tensor. For example, we can compare the inferred gene-gene interactions with known interactions from databases like TRRUST or with results from other GRN inference methods like SCENIC+. We can examine specific time slices 𝓙(*t*) to see if they recapitulate known regulatory events at that stage of the biological process (e.g., the activation of key lineage-defining transcription factors).

With the validated temporal Jacobian tensor in hand, we are ready for the final and most interpretive step of the core framework: decomposing this complex object into a small number of fundamental, time-independent regulatory programs. This is the subject of the next chapter.
'''

# Chapter 7: Regulatory Archetype Extraction

The temporal Jacobian tensor, 𝓙 ∈ ℝ<sup>G×G×T</sup>, provides a complete, time-varying description of the gene regulatory network. However, its high dimensionality makes it difficult to interpret directly. A *G* × *G* matrix at each of *T* time points represents a vast amount of information. The key to unlocking the biological meaning hidden within this tensor is to find the underlying patterns—the recurring motifs of regulation that are active at different times and in different combinations. In scQDiff, we call these fundamental patterns **regulatory archetypes**. This chapter describes how we use tensor decomposition to extract these archetypes and their temporal activation profiles.

## 7.1 The Goal: Finding Low-Rank Structure

The central hypothesis is that the complex, time-varying GRN is not arbitrary. Instead, it is constructed from a relatively small number of core regulatory programs or modules. For example, in T-cell activation, there might be an "early response" archetype involving inflammatory signaling, a "proliferation" archetype involving cell cycle genes, and a "differentiation" archetype involving lineage-specific transcription factors. The observed GRN at any given time is a mixture of these archetypes, with their relative importance changing as the cell progresses through the activation process.

Mathematically, this hypothesis translates to the assumption that the temporal Jacobian tensor is **low-rank**. This means that it can be well-approximated by the sum of a small number of simpler, rank-one tensors. A rank-one tensor is the outer product of three vectors (or in our case, two matrices and one vector). This is the principle behind **tensor decomposition**.

## 7.2 Tensor Decomposition via Singular Value Decomposition (SVD)

To perform the decomposition, we first "unfold" or "matricize" the tensor 𝓙. We reshape the *G* × *G* × *T* tensor into a (*G* × *G*) × *T* matrix, where each column represents the flattened Jacobian matrix at a single time point. Let this unfolded matrix be denoted **J**<sub>unfold</sub>.

We then perform **Singular Value Decomposition (SVD)** on this matrix. SVD is a fundamental matrix factorization technique that decomposes any matrix **A** into the product of three other matrices: **A** = **U** **Σ** **V**<sup>T</sup>, where:
-   **U** is an orthogonal matrix whose columns are the left singular vectors.
-   **Σ** is a diagonal matrix whose diagonal entries are the singular values, *σ<sub>k</sub>*.
-   **V**<sup>T</sup> is an orthogonal matrix whose rows are the right singular vectors.

For our unfolded Jacobian matrix, the SVD gives us:

$$\mathbf{J}_{\text{unfold}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

The best rank-*K* approximation of **J**<sub>unfold</sub> is obtained by keeping only the top *K* singular values and their corresponding singular vectors:

$$\mathbf{J}_{\text{unfold}} \approx \sum_{k=1}^{K} \sigma_k \mathbf{u}_k \mathbf{v}_k^T$$

Now, we can "refold" this approximation back into the original tensor shape. The left singular vectors **u**<sub>k</sub>, which have dimension *G* × *G*, are reshaped back into *G* × *G* matrices. These are our **regulatory archetypes**, *M<sub>k</sub>*. The right singular vectors **v**<sub>k</sub>, which have dimension *T*, combined with the singular values *σ<sub>k</sub>*, give us the **temporal activation profiles**, *a<sub>k</sub>*(*t*).

The final decomposition is:

$$\mathcal{J}(t) \approx \sum_{k=1}^{K} a_k(t) M_k$$

Where:
-   *M<sub>k</sub>* ∈ ℝ<sup>G×G</sup> is the *k*-th **regulatory archetype**. It is a time-independent matrix representing a fundamental mode of gene-gene interaction.
-   *a<sub>k</sub>*(*t*) is the **temporal activation profile** of the *k*-th archetype. It is a scalar function of time that describes when and how strongly each archetype is active.

This decomposition provides a compact and powerful summary of the GRN dynamics. Instead of analyzing thousands of time-varying matrices, we now have a small number of static archetypes and their corresponding time-dependent activation levels.

## 7.3 Biological Interpretation of Archetypes

Each archetype matrix *M<sub>k</sub>* represents a specific regulatory program. To interpret it, we can analyze its structure:

-   **Identify Key Regulators**: The rows of *M<sub>k</sub>* correspond to target genes, and the columns correspond to regulators. A column with many large entries (positive or negative) corresponds to a "hub" regulator in that archetype—a gene that influences many other genes within that program.
-   **Identify Key Targets**: A row with many large entries corresponds to a gene that is regulated by many other genes within the archetype.
-   **Enrichment Analysis**: We can perform gene set enrichment analysis (GSEA) on the genes that are most prominent in an archetype (either as regulators or targets) to identify the biological pathways or processes associated with that program (e.g., "interferon response," "cell cycle progression").

## 7.4 Determining the Number of Archetypes (*K*)

The choice of *K*, the number of archetypes to extract, is a critical modeling decision. A *K* that is too small will fail to capture the full complexity of the dynamics, while a *K* that is too large will lead to overfitting and redundant, uninterpretable archetypes.

There are several heuristics for choosing *K*:

-   **Scree Plot**: We can plot the singular values *σ<sub>k</sub>* in descending order. The "elbow" of this plot, where the singular values start to level off, is often a good choice for *K*. This indicates the point of diminishing returns, where additional archetypes explain very little of the variance in the data.
-   **Reconstruction Error**: We can choose *K* such that the reconstruction error, ||𝓙 - Σ *a<sub>k</sub>*(*t*)*M<sub>k</sub>*||², is below a certain threshold (e.g., captures 80% of the variance).
-   **Biological Interpretability**: We can try several values of *K* and choose the one that yields the most distinct and biologically interpretable archetypes. If two archetypes are highly correlated or represent very similar biological processes, it may be a sign that *K* is too large.

## 7.5 Validation Against Known Regulons

To validate the biological relevance of the learned archetypes, we can compare them to known regulatory networks. For example, we can use a tool like SCENIC+ to infer transcription factor regulons from the scRNA-seq and scATAC-seq data. We can then check if the interactions within a learned archetype *M<sub>k</sub>* are enriched for the known targets of a specific transcription factor. A strong correspondence between a data-driven archetype and a literature-curated or motif-based regulon provides strong evidence for the biological validity of the scQDiff results.

With the GRN dynamics now distilled into a set of interpretable archetypes, we are equipped to perform the ultimate tasks of the scQDiff framework: predicting cellular behavior under new conditions and designing targeted interventions. The next chapters will show how these archetypes are used for guided trajectory synthesis and the calculation of control energy.

# Chapter 8: Guided Trajectory Synthesis

Having learned the endogenous drift field and decomposed it into interpretable regulatory archetypes, we now move from a descriptive to a predictive and prescriptive framework. **Guided Trajectory Synthesis** is the process of simulating cellular trajectories under the influence of external controls. This allows us to ask "what if?" questions: What if we activate a specific regulatory program? What if we try to reverse a disease progression? This chapter details the mathematical formulation of guided synthesis, its implementation for both forward prediction and reverse engineering, and the role of archetypes in constructing meaningful interventions.

## 8.1 The Controlled Stochastic Differential Equation

The core idea of guided synthesis is to introduce an external **control input**, **u**(**x**, *t*), into the baseline stochastic differential equation (SDE). The dynamics are no longer governed solely by the endogenous drift *f* but by a combination of the natural dynamics and our desired intervention.

The **controlled SDE** is:

$$d\mathbf{x}_t = [f(\mathbf{x}_t, t) + \mathbf{u}(\mathbf{x}_t, t)] \, dt + \sqrt{2\varepsilon} \, d\mathbf{W}_t$$

Here, **u**(**x**, *t*) represents the perturbation we are applying. It could be the effect of a drug, the forced overexpression of a gene, or a more complex, time-varying intervention. The goal of guided synthesis is to design a meaningful control input **u** and then simulate the resulting trajectories by integrating this SDE over time.

## 8.2 Archetype-Based Guidance

How should we design the control input **u**? A naive approach might be to simply push the cell in the direction of a target state. However, this is biologically unrealistic and inefficient. A cell is not a simple particle; it is a complex system with preferred modes of response. A more effective strategy is to design interventions that work *with* the cell's own regulatory machinery.

This is where the regulatory archetypes become invaluable. Each archetype *M<sub>k</sub>* represents a coordinated regulatory program that the cell "knows" how to execute. We can construct a powerful and biologically plausible control input by creating a linear combination of these archetype vector fields:

$$\mathbf{u}(\mathbf{x}, t) = \gamma \sum_{k=1}^{K} g_k(t) M_k \mathbf{x}$$

Where:
-   *γ* is a scalar **guidance strength** parameter, controlling the overall magnitude of the intervention.
-   *g<sub>k</sub>*(*t*) is the **guidance profile** for archetype *k*, a user-defined function of time that specifies when and how strongly we want to activate or repress that particular regulatory program.

This formulation allows us to design highly specific interventions. For example, to promote differentiation into a specific lineage, we could up-regulate the guidance profile *g<sub>k</sub>*(*t*) for the archetype *M<sub>k</sub>* that corresponds to that lineage's master regulators.

## 8.3 Forward Prediction: Simulating Perturbations

In **forward prediction**, we start with an initial cell state (or distribution of states) and simulate how it will evolve under a specific intervention. This is useful for predicting the effect of a drug or a genetic perturbation.

**Example**: Suppose we want to predict the effect of a drug that is known to activate a specific signaling pathway. We first identify the archetype *M<sub>drug</sub>* that is most enriched for genes in that pathway. We can then design a guidance profile *g<sub>drug</sub>*(*t*) that mimics the drug's effect (e.g., a constant positive value) and set all other *g<sub>k</sub>*(*t*) to zero. The control input becomes **u**(**x**, *t*) = *γ* *g<sub>drug</sub>*(*t*) *M<sub>drug</sub>* **x**.

To perform the simulation:

1.  **Initialize**: Start with a cell at state **x**<sub>0</sub> at *t* = 0.
2.  **Iterate**: For each small time step Δ*t*, update the cell's state using the Euler-Maruyama method for SDEs:
    $$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + [f(\mathbf{x}_t, t) + \mathbf{u}(\mathbf{x}_t, t)] \Delta t + \sqrt{2\varepsilon \Delta t} \, \mathbf{Z}$$
    where **Z** is a vector of independent standard normal random variables.
3.  **Terminate**: Continue until *t* = 1. The resulting path {**x**<sub>t</sub>} is a simulated trajectory under the drug's influence.

By simulating many such trajectories from a distribution of initial states, we can predict the population-level response to the drug.

## 8.4 Reverse Engineering: Designing Reprogramming Strategies

Perhaps the most exciting application of guided synthesis is **reverse engineering**, where we aim to steer a cell from an undesirable state (e.g., a cancer cell) to a desirable one (e.g., a healthy cell or an apoptotic state). The goal is to find a control input **u** that achieves this transition.

In scQDiff, we achieve this by running the system's dynamics in reverse. Suppose we have learned the endogenous dynamics of a process, such as cancer progression, from *t* = 0 (early stage) to *t* = 1 (late stage). To design a therapy that reverses this process, we can guide the cell with the *time-reversed* endogenous dynamics.

The guidance profile for reverse engineering is chosen to be the negative of the time-reversed activation profiles of the endogenous archetypes:

$$g_k(t) = -a_k(1-t)$$

This means that at the beginning of our reprogramming simulation (*t* = 0), we apply a control that strongly counteracts the archetypes that were active at the *end* of the disease progression (*t* = 1). As the simulation proceeds, we gradually shift to counteracting the earlier disease archetypes. The control input becomes:

$$\mathbf{u}(\mathbf{x}, t) = -\gamma \sum_{k=1}^{K} a_k(1-t) M_k \mathbf{x}$$

This is a powerful concept: the optimal way to reverse a biological process is to systematically undo the sequence of regulatory programs that created it, in reverse order. This provides a principled, data-driven strategy for designing complex, time-varying therapeutic interventions.

## 8.5 Uncertainty Quantification

Because the dynamics are stochastic (due to the diffusion term), each simulation of a guided trajectory will produce a slightly different path. This is a feature, not a bug. It reflects the inherent stochasticity of biological systems. By running the same guided synthesis simulation many times (e.g., 100 times) from the same starting cell, we can generate a **distribution of possible outcomes**. We can then quantify the uncertainty of our prediction:

-   **Mean Trajectory**: The average path taken by the simulated cells.
-   **Confidence Cone**: The region of state space that contains, for example, 95% of the simulated trajectories at each time point.

This provides a measure of how robust the predicted outcome is. A narrow confidence cone suggests a reliable, deterministic outcome, while a wide cone suggests that the intervention may lead to a heterogeneous population of cells with different fates.

With the ability to both predict and prescribe cellular trajectories, we need a way to quantify the cost or difficulty of these interventions. This is the role of control energy, which we will explore in detail in the next chapter.

# Chapter 9: Control Energy Theory and Applications

Guided trajectory synthesis allows us to design interventions to steer cells towards desired states, but it does not tell us how difficult or biologically feasible these interventions are. Is it easier to turn a fibroblast into a neuron or a cardiomyocyte? Is reversing a specific cancer type a realistic goal or a Sisyphean task? To answer such questions, we need a quantitative measure of the "effort" required for cellular reprogramming. In scQDiff, this is provided by the concept of **Control Energy**, a metric derived from optimal control theory that is deeply connected to the learned regulatory archetypes. This chapter provides the full mathematical derivation of control energy and illustrates its application in assessing the feasibility of cellular transitions.

## 9.1 Mathematical Derivation of Control Energy

As introduced in Chapter 8, the dynamics of a cell under an external intervention **u**(**x**, *t*) are described by the controlled SDE:

$$d\mathbf{x}_t = [f(\mathbf{x}_t, t) + \mathbf{u}(\mathbf{x}_t, t)] \, dt + \sqrt{2\varepsilon} \, d\mathbf{W}_t$$

The control input **u** represents the force we must apply to the cell to deviate it from its natural trajectory, which is governed by the endogenous drift *f*. Intuitively, the "harder" we have to push the cell, the more difficult the transition is.

Optimal control theory provides a way to formalize this intuition. The **control energy** is defined as the total integrated squared magnitude of the control input required to drive the system along a specific path {**x**<sub>t</sub>} from *t* = 0 to *t* = 1:

$$E_{\text{control}} = \int_0^1 \|\mathbf{u}(\mathbf{x}_t, t)\|_2^2 \, dt$$

This definition has a clear physical interpretation: it is the total work done by the external control force over the entire trajectory. A high control energy implies that a large, sustained intervention is necessary, suggesting the transition is biologically difficult, energetically costly, or "irreversible." A low control energy implies that the transition is relatively easy to achieve, perhaps requiring only a small, transient perturbation.

## 9.2 Connection to Archetypes and Guided Synthesis

The power of this concept becomes fully apparent when we connect it to the archetype-based guidance introduced in the previous chapter. Recall that our control input is constructed as a weighted combination of the regulatory archetype vector fields:

$$\mathbf{u}(\mathbf{x}, t) = \gamma \sum_{k=1}^{K} g_k(t) M_k \mathbf{x}$$

Substituting this into the control energy definition, we arrive at the final formula for the control energy of a specific reprogramming trajectory {**x**<sub>t</sub>}:

$$\boxed{E_{\text{control}} = \int_0^1 \left\| \gamma \sum_{k=1}^{K} g_k(t) M_k \mathbf{x}_t \right\|_2^2 \, dt}$$

This is the central result of this chapter. It provides a direct, computable link between the difficulty of a cellular transition and the underlying regulatory architecture of the cell. The control energy depends on:

-   The **guidance strength** *γ*: A stronger push requires more energy.
-   The **guidance profiles** *g<sub>k</sub>*(*t*): The specific sequence of archetype activations/repressions.
-   The **archetype matrices** *M<sub>k</sub>*: The strength and structure of the endogenous regulatory programs.
-   The **trajectory** **x**<sub>t</sub>: The specific path the cell takes through gene expression space.

## 9.3 Cellular Reprogramming Example: Step-by-Step Calculation

Let's revisit the simplified 2-gene reprogramming example from Chapter 8 to illustrate the calculation.

-   **System**: A 2-gene system with an oncogene (*x*₁) and an effector gene (*x*₂).
-   **States**: Diseased **x**<sub>D</sub> = [1, 0]<sup>T</sup> and Healthy **x**<sub>H</sub> = [0, 1]<sup>T</sup>.
-   **Archetype**: A single repressive archetype *M*₁ = [[0, 0], [-2, 0]].
-   **Guidance**: Reverse synthesis with *g*₁(*t*) = -1 and *γ* = 1.

**Step 1: Define the Trajectory**

We first need the path {**x**<sub>t</sub>} that the cell follows under the guided dynamics. For this simple example, we can approximate this as a straight line in gene space:

$$\mathbf{x}_t = \mathbf{x}_D + t(\mathbf{x}_H - \mathbf{x}_D) = \begin{pmatrix} 1 \\ 0 \end{pmatrix} + t \begin{pmatrix} -1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1-t \\ t \end{pmatrix}$$

In a real application, this path would be obtained by simulating the controlled SDE as described in Chapter 8.

**Step 2: Calculate the Control Input Along the Trajectory**

We evaluate the control input **u** at each point along the trajectory **x**<sub>t</sub>:

$$\mathbf{u}(\mathbf{x}_t, t) = (1)(-1) M_1 \mathbf{x}_t = - \begin{pmatrix} 0 & 0 \\ -2 & 0 \end{pmatrix} \begin{pmatrix} 1-t \\ t \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 2 & 0 \end{pmatrix} \begin{pmatrix} 1-t \\ t \end{pmatrix} = \begin{pmatrix} 0 \\ 2(1-t) \end{pmatrix}$$

**Step 3: Calculate the Squared Magnitude**

Next, we find the squared L2-norm of this control vector at each time *t*:

$$\|\mathbf{u}(\mathbf{x}_t, t)\|_2^2 = 0^2 + (2(1-t))^2 = 4(1-t)^2$$

This term, 4(1-*t*)², represents the **instantaneous control power** being applied at time *t*. Note that the power is highest at *t* = 0 (when the oncogene level is highest) and decreases to zero as the cell approaches the healthy state at *t* = 1.

**Step 4: Integrate Over Time**

Finally, we integrate this instantaneous power over the entire trajectory from *t* = 0 to *t* = 1 to get the total control energy:

$$E_{\text{control}} = \int_0^1 4(1-t)^2 \, dt = 4 \left[ -\frac{(1-t)^3}{3} \right]_0^1 = 4 \left( 0 - \left(-\frac{1}{3}\right) \right) = \frac{4}{3}$$

The total control energy required for this reprogramming is 4/3. This single number summarizes the difficulty of the entire process. We can now use it to compare different potential reprogramming strategies or to assess the relative stability of different disease states.

## 9.4 Irreversibility and Round-Trip Error

Control energy provides a natural way to quantify the concept of **irreversibility** in biological processes. A transition is considered highly irreversible if the energy required to reverse it is very high.

We can define a **Round-Trip Error** to measure this. First, we simulate a forward process (e.g., disease progression) from a healthy state **x**<sub>H</sub> to a diseased state **x**<sub>D</sub>. Then, we use reverse engineering to simulate the reprogramming trajectory from **x**<sub>D</sub> back towards **x**<sub>H</sub>. The final state of the reversed trajectory, **x**<sub>H</sub>\[\]', may not be identical to the original healthy state.

The Round-Trip Error can be defined as the Euclidean distance between the original and final healthy states, ||**x**<sub>H</sub> - **x**<sub>H</sub>\[\]'\|₂. A large round-trip error, coupled with a high control energy for the reverse path, is a strong indicator that the biological process is a stable, quasi-irreversible transition, such as terminal differentiation or oncogenic transformation.

This concludes Part II of the booklet. We have now covered the entire core scQDiff framework, from learning the drift field to extracting interpretable archetypes and using them for predictive control. In Part III, we will explore how to extend this powerful framework to handle the increasing complexity of modern single-cell datasets, including multi-omic and spatial data.
'''# Chapter 10: Multi-Omic Integration

Single-cell RNA sequencing provides a powerful window into the transcriptional state of a cell, but it is only one piece of the regulatory puzzle. Cellular identity and function are governed by a complex interplay of multiple molecular layers, from the epigenetic landscape of chromatin accessibility to the abundance of functional proteins. The rise of multi-modal single-cell technologies, which measure two or more of these layers simultaneously from the same cell (e.g., sc-multiome for RNA + ATAC, CITE-seq for RNA + Protein), presents both a challenge and an opportunity. This chapter explores how the scQDiff framework can be extended to integrate these rich, multi-omic datasets, moving towards a truly holistic model of cellular regulation.

## 10.1 The Need for a Multi-Layered View

Gene regulation is not a linear process. The transcription of a gene is controlled by the accessibility of its promoter and enhancer regions (the epigenetic layer, measured by scATAC-seq), which in turn is controlled by the binding of transcription factors (TFs). The resulting mRNA transcript (measured by scRNA-seq) is then translated into a protein, whose activity can be modulated by post-translational modifications and interactions with other proteins (the proteomic layer, partially measured by CITE-seq). A complete model of cellular dynamics must account for these cross-modal interactions.

For example, to understand how a TF activates a target gene, we need to model not just the TF's expression level but also the accessibility of the target gene's enhancer. The effect of the TF on the target gene's expression rate is *conditional* on the chromatin state. A model based solely on RNA expression cannot capture this crucial dependency.

## 10.2 Mode 1: The Unified Multi-Modal Tensor

The most direct way to extend scQDiff is to treat features from different modalities as components of a single, unified state vector. We can concatenate the feature vectors from each modality into a larger state vector, **x'**:

$$\mathbf{x}' = [\mathbf{x}_{\text{RNA}}, \mathbf{x}_{\text{ATAC}}, \mathbf{x}_{\text{Protein}}]$$

We can then apply the standard scQDiff pipeline to this unified state vector. We learn a single drift field *f*(*x'*, *t*) and compute its Jacobian, ∇*f*. This unified Jacobian will be a larger matrix that can be partitioned into blocks, each representing a specific type of cross-modal interaction:

$$J = \begin{pmatrix}
J_{\text{RNA,RNA}} & J_{\text{RNA,ATAC}} & J_{\text{RNA,Protein}} \\
J_{\text{ATAC,RNA}} & J_{\text{ATAC,ATAC}} & J_{\text{ATAC,Protein}} \\
J_{\text{Protein,RNA}} & J_{\text{Protein,ATAC}} & J_{\text{Protein,Protein}}
\end{pmatrix}$$

The off-diagonal blocks are of particular interest:

-   **J<sub>RNA,ATAC</sub>**: This block, with entries ∂*f*<sub>gene</sub>/∂*a*<sub>peak</sub>, quantifies the effect of chromatin accessibility at a specific peak on the expression rate of a gene. This directly models the dynamic activity of cis-regulatory elements.
-   **J<sub>RNA,Protein</sub>**: This block, with entries ∂*f*<sub>gene</sub>/∂*p*<sub>protein</sub>, quantifies how the abundance of a protein (e.g., a transcription factor) affects the expression rate of a gene, providing a more direct measure of trans-regulation than using the TF's mRNA level.
-   **J<sub>ATAC,RNA</sub>**: This block could model how the expression of certain genes (e.g., chromatin remodelers) affects the accessibility of the chromatin landscape.

**Advantages:**
-   Conceptually simple and requires minimal changes to the core scQDiff pipeline.
-   Directly models all possible cross-modal interactions.

**Limitations:**
-   **Scalability**: The size of the unified state vector can become very large, posing computational challenges.
-   **Feature Heterogeneity**: Different modalities have very different statistical properties (e.g., extreme sparsity of scATAC-seq data), which may require specialized normalization and noise models.

## 10.3 Mode 2: Coupled Stochastic Differential Equations

A more sophisticated approach is to model each modality with its own SDE, but to couple them through their drift terms. This allows for more flexible modeling of the interactions and noise properties of each data type. For a system with RNA and ATAC data, the model would be:

$$d\mathbf{x}_{\text{RNA}} = f_{\text{RNA}}(\mathbf{x}_{\text{RNA}}, \mathbf{x}_{\text{ATAC}}, t) \, dt + \sqrt{2\varepsilon_{\text{RNA}}} \, d\mathbf{W}_{t, \text{RNA}}$$
$$d\mathbf{x}_{\text{ATAC}} = f_{\text{ATAC}}(\mathbf{x}_{\text{RNA}}, \mathbf{x}_{\text{ATAC}}, t) \, dt + \sqrt{2\varepsilon_{\text{ATAC}}} \, d\mathbf{W}_{t, \text{ATAC}}$$

Here, the drift for the RNA modality, *f*<sub>RNA</sub>, depends on both the RNA state and the ATAC state. This explicitly models the fact that gene expression dynamics are conditional on the chromatin landscape. Similarly, the drift for the ATAC modality, *f*<sub>ATAC</sub>, can depend on the RNA state, capturing feedback loops where expressed genes can alter chromatin structure.

Learning these coupled drift fields is more complex, likely requiring a multi-task learning setup where two neural networks are trained jointly to predict the dynamics of both modalities simultaneously. The resulting Jacobians would be computed separately for each drift field, but their interpretation would be linked.

**Advantages:**
-   More flexible and biologically realistic modeling of cross-modal dependencies.
-   Allows for different noise models (ε<sub>RNA</sub> vs. ε<sub>ATAC</sub>) for each modality.

**Limitations:**
-   Increased model complexity and computational cost.
-   Requires careful design of the neural network architecture and training procedure.

## 10.4 Mode 3: Anchored Decomposition with Multi-Omic Priors

This mode focuses on using multi-omic data to enhance the interpretability of the learned archetypes. Even if we only model the RNA dynamics, we can use information from other modalities to guide the tensor decomposition step. This is known as **anchored decomposition**.

Tools like **SCENIC+** are specifically designed to integrate scRNA-seq and scATAC-seq to build more accurate gene regulatory networks. SCENIC+ identifies transcription factor regulons and links them to specific cis-regulatory elements (enhancers) based on motif analysis and co-variation of expression and accessibility. The output is a high-confidence, prior GRN.

We can use this prior network to constrain the structure of the regulatory archetypes *M<sub>k</sub>* during the SVD decomposition. For example, we can add a regularization term to the decomposition objective that penalizes non-zero entries in *M<sub>k</sub>* that do not correspond to a known interaction in the SCENIC+ network. This ensures that the learned archetypes are biologically grounded and consistent with the known cis- and trans-regulatory logic.

Similarly, for protein data, prior networks from databases or tools like **PINNACLE** (which infers protein-protein interaction networks) can be used to anchor the decomposition of a protein-level Jacobian.

**Advantages:**
-   Greatly enhances the biological interpretability and confidence in the learned archetypes.
-   Reduces the risk of learning spurious correlations by incorporating domain knowledge.

**Limitations:**
-   Depends on the availability and quality of the prior networks.
-   Requires a more complex, regularized tensor decomposition algorithm.

These three modes provide a roadmap for extending scQDiff into a comprehensive multi-omic framework. By unifying modalities, coupling their dynamics, or anchoring their interpretation, we can move towards a model that captures the full, multi-layered complexity of cellular regulation. The next chapter will explore another critical dimension: space.
'''

# Chapter 11: Spatial Tensor Fields

Cells do not exist in a vacuum. Within a tissue, their behavior is profoundly influenced by their spatial location and their interactions with neighboring cells. The tissue microenvironment—comprising the extracellular matrix, signaling gradients, and direct cell-cell contacts—creates a complex spatial landscape that shapes gene regulatory networks. The emergence of spatial transcriptomics technologies, which measure gene expression while preserving spatial coordinates, provides an opportunity to integrate this critical dimension into our models of cellular dynamics. This chapter explores how the scQDiff framework can be extended to learn **spatial tensor fields**, moving from a single, global model of regulation to a spatially-heterogeneous one.

## 11.1 The Importance of Spatial Context

In many biological processes, space is not a passive backdrop but an active participant. During embryonic development, morphogen gradients establish body axes and define distinct developmental fields where cells adopt different fates. In the immune system, the spatial organization of lymph nodes is critical for orchestrating immune responses. In cancer, the tumor microenvironment, including its spatial arrangement of cancer cells, stromal cells, and immune cells, plays a key role in tumor growth, metastasis, and drug resistance.

Traditional single-cell RNA-seq, which requires dissociating tissues into a suspension of single cells, destroys this spatial information. Spatial transcriptomics methods, such as 10x Visium, Slide-seq, and MERFISH, overcome this limitation by measuring gene expression in situ. The result is a dataset where each cell (or small group of cells) has both a gene expression profile **x** and a set of spatial coordinates **s** = (*s*<sub>x</sub>, *s*<sub>y</sub>).

## 11.2 Spatially-Varying Drift and Jacobian

To incorporate spatial information into scQDiff, we allow the drift field and, consequently, the Jacobian tensor to become functions of physical space. The core SDE of the model is modified to:

$$d\mathbf{x}_t = f(\mathbf{x}_t, t, \mathbf{s}) \, dt + \sqrt{2\varepsilon} \, d\mathbf{W}_t$$

Here, the drift field *f* now depends on the spatial coordinates **s** in addition to the cell state **x** and pseudotime *t*. This means that two cells with identical gene expression profiles but located in different parts of the tissue can have different dynamics—they are subject to different regulatory forces.

Correspondingly, the Jacobian tensor also becomes a function of space:

$$\mathcal{J}(t, \mathbf{s}) = \nabla_x f(\mathbf{x}, t, \mathbf{s})$$

This object, 𝓙(*t*, **s**), is a **spatial tensor field**. At each point in space and time, it provides the full *G* × *G* matrix of gene-gene influences. This allows us to ask questions like: How does the regulatory network of a cancer cell at the invasive front of a tumor differ from that of a cell in the tumor core? How do signaling gradients from a niche of stromal cells alter the regulatory archetypes active in nearby stem cells?

## 11.3 Learning the Spatial Field

Learning the spatially-varying drift field *f*(*x*, *t*, **s**) requires a modification of the methods described in Chapter 5. The neural network used to parameterize the drift, *f*<sub>θ</sub>, must now take the spatial coordinates **s** as an additional input.

$$f_{\theta}(\mathbf{x}, t, \mathbf{s})$$

During training, for each cell *i*, we provide its gene expression vector **x**<sub>i</sub>, its pseudotime *t<sub>i</sub>*, and its spatial coordinates **s**<sub>i</sub>. The network then learns to predict the cell's RNA velocity (in the velocity-constrained approach) or its displacement in latent space (in the Bridge-Lite approach) based on all three inputs. The network will learn to represent the spatial heterogeneity in the dynamics.

## 11.4 Applications and Interpretation

Once the spatial tensor field 𝓙(*t*, **s**) is learned, it opens up a range of new analyses:

-   **Visualizing Spatial GRNs**: We can create maps of the tissue showing how specific regulatory interactions (*J<sub>ij</sub>*) vary across space. For example, we could visualize the strength of a key signaling pathway by plotting the magnitude of the corresponding Jacobian entries at each spatial location.

-   **Identifying Spatial Archetypes**: We can perform tensor decomposition on the spatial tensor field to identify spatial archetypes. This might involve decomposing the 4D tensor 𝓙(*t*, **s**) ∈ ℝ<sup>G×G×T×S</sup> (where S is the number of spatial locations) or performing separate decompositions in different spatial regions and comparing the resulting archetypes.

-   **Modeling Cell-Cell Communication**: The spatial variation in the Jacobian can be correlated with the expression of ligands and receptors in neighboring cells. For example, if we observe that the Jacobian of epithelial cells changes in regions with high expression of the ligand Wnt by nearby stromal cells, this provides evidence for a Wnt-mediated signaling interaction that modulates the epithelial GRN.

-   **Spatially-Targeted Interventions**: Guided trajectory synthesis can be made space-dependent. We could design an intervention **u**(**x**, *t*, **s**) that is only active in a specific region of the tissue, allowing for the design of more precise therapies that target diseased cells while sparing healthy ones.

## 11.5 Challenges and Future Directions

The integration of spatial data into scQDiff is a frontier with several challenges:

-   **Scalability**: Spatial transcriptomics datasets can be massive, containing hundreds of thousands of spatial locations. Learning a high-dimensional function over this space is computationally demanding.
-   **Resolution**: The resolution of spatial methods varies. Some methods provide single-cell resolution, while others average over small groups of cells. The model must be able to handle this variation.
-   **Data Integration**: Combining time-series data (which often requires dissociating tissues) with spatial data (which is often from a single time point) is a major experimental and computational challenge. Methods for aligning and integrating these different data types are an active area of research.

Despite these challenges, the extension of scQDiff to spatial tensor fields represents a major step towards a truly comprehensive model of cellular dynamics—one that accounts for not only the internal regulatory state of the cell but also its position and interactions within the complex ecosystem of the tissue. The next chapter will discuss a final mode of extension: using prior biological knowledge to anchor the decomposition of the Jacobian tensor, thereby enhancing its interpretability and biological relevance.

# Chapter 12: Anchored Decomposition with Prior Networks

The tensor decomposition described in Chapter 7 is a purely data-driven method for discovering regulatory archetypes. While powerful, it can sometimes yield archetypes that are difficult to interpret or that do not align perfectly with our existing biological knowledge. A more robust and interpretable approach is to guide or "anchor" the decomposition using prior information about the gene regulatory network. This chapter details how scQDiff can be extended to incorporate prior networks from tools like SCENIC+ and PINNACLE, leading to the discovery of more biologically grounded and validated regulatory archetypes.

## 12.1 The Value of Prior Knowledge

The field of systems biology has spent decades curating databases of known gene and protein interactions. More recently, computational tools have been developed to infer these interactions from multi-omic data with increasing accuracy. For example:

-   **SCENIC+** integrates scRNA-seq and scATAC-seq to identify high-confidence transcription factor (TF) regulons. It uses TF binding motifs in accessible chromatin regions to link TFs to their direct target genes.
-   **PINNACLE** infers protein-protein interaction (PPI) networks and signaling pathways from single-cell proteomic or transcriptomic data.
-   **Public Databases** like TRRUST, STRING, and Reactome contain vast amounts of curated information about TF-target interactions, PPIs, and metabolic pathways.

Ignoring this wealth of information would be a missed opportunity. By integrating it into the scQDiff framework, we can achieve several goals:

1.  **Enhance Interpretability**: Ensure that the learned archetypes correspond to known biological pathways or regulons.
2.  **Improve Robustness**: Reduce the risk of learning spurious, non-causal correlations from noisy single-cell data.
3.  **Generate Testable Hypotheses**: When the data-driven dynamics suggest a modification to a known pathway, it generates a strong, testable hypothesis for experimental validation.

## 12.2 Regularized Tensor Decomposition

The standard SVD-based decomposition finds the best low-rank approximation of the temporal Jacobian tensor in a purely mathematical sense (minimizing the Frobenius norm of the reconstruction error). To incorporate prior knowledge, we modify this objective by adding a **regularization term**. This term penalizes solutions that are inconsistent with the prior network.

The new objective is to find the archetypes *M<sub>k</sub>* and activations *a<sub>k</sub>*(*t*) that minimize:

$$\left\| \mathcal{J}(t) - \sum_{k=1}^{K} a_k(t) M_k \right\|_F^2 + \lambda \sum_{k=1}^{K} \Omega(M_k, \mathbf{P})$$

Where:
-   The first term is the standard reconstruction error.
-   Ω(*M<sub>k</sub>*, **P**) is the regularization function, which measures the inconsistency between archetype *M<sub>k</sub>* and the prior network **P**.
-   *λ* is a hyperparameter that controls the strength of the regularization—how much we trust the prior knowledge versus the data.

## 12.3 Designing the Regularization Function

The choice of the regularization function Ω depends on the nature of the prior network **P**. Let **P** be a *G* × *G* adjacency matrix where *P<sub>ij</sub>* = 1 if gene *j* is known to regulate gene *i*, and 0 otherwise.

A simple and effective regularizer is a weighted L1 norm:

$$\Omega(M_k, \mathbf{P}) = \sum_{i,j} W_{ij} |M_{k,ij}|$$

Where *W<sub>ij</sub>* is a weight matrix. We can set the weights to encourage the archetypes to conform to the prior network. For example:

-   If *P<sub>ij</sub>* = 0 (no known interaction), set *W<sub>ij</sub>* to a high value. This penalizes non-zero entries in the archetype matrix where no interaction is expected, effectively acting as a sparsity constraint that encourages the model to only learn interactions that are either known or strongly supported by the data.
-   If *P<sub>ij</sub>* = 1 (a known interaction exists), set *W<sub>ij</sub>* to a low value (or zero). This allows the model to freely learn the strength and sign of the interaction from the data.

This approach, known as **network-regularized matrix factorization**, gently guides the decomposition towards solutions that are consistent with our prior knowledge, without rigidly enforcing it. The model is still free to discover novel interactions if the evidence in the data is strong enough to overcome the penalty.

## 12.4 The BRIDGE Framework: A Practical Implementation

The integration of SCENIC+ and PINNACLE outputs with the scQDiff framework is a key implementation, which we refer to as **BRIDGE**. The BRIDGE framework does not require running the SCENIC+ or PINNACLE software itself, but rather uses their output files as the prior network **P**.

The workflow is as follows:

1.  **Run SCENIC+**: On the scRNA-seq and scATAC-seq data, generate a high-confidence GRN based on TF motifs and expression-accessibility correlation.
2.  **Run scQDiff (Core)**: Learn the temporal Jacobian tensor 𝓙(*t*) from the time-series scRNA-seq data as described in Part II.
3.  **Perform Anchored Decomposition**: Use the SCENIC+ network as the prior **P** to perform the regularized tensor decomposition of 𝓙(*t*). This yields a set of archetypes that are explicitly linked to the activity of specific transcription factors and their target genes.
4.  **Interpret Archetypes**: Each archetype can now be directly interpreted in terms of the TF regulons it represents. For example, Archetype 1 might correspond to the SPI1 regulon, while Archetype 2 corresponds to the GATA1 regulon.

This anchored approach transforms the archetypes from abstract mathematical objects into concrete, biologically meaningful regulatory modules with clear mechanistic interpretations.

## 12.5 Benefits and Summary of Extensions

Anchored decomposition completes our tour of the extensions to the core scQDiff framework. By integrating multi-omic data, spatial coordinates, and prior biological knowledge, we can build a model of cellular dynamics that is far more comprehensive, robust, and interpretable than one based on transcriptomic data alone.

-   **Multi-Omics** (Chapter 10) provides a deeper, multi-layered view of the regulatory state.
-   **Spatial Fields** (Chapter 11) accounts for the crucial role of the tissue microenvironment.
-   **Anchored Decomposition** (Chapter 12) grounds the learned dynamics in decades of accumulated biological knowledge.

Together, these extensions position scQDiff as a flexible and powerful platform for integrative single-cell data science. The final part of this booklet will shift from theory to practice, providing detailed worked examples, experimental validation designs, and a guide to the computational implementation of the framework.

# Chapter 13: Worked Example: T-Cell Activation

To demonstrate the full power of the scQDiff framework, we will now walk through a complete, end-to-end analysis of a real biological process: the activation of human T-cells. T-cell activation is a cornerstone of the adaptive immune response and a well-studied model system for cellular differentiation and decision-making. It involves a complex cascade of gene expression changes as naive T-cells transition into activated effector cells over the course of several days. This chapter will serve as a practical guide, illustrating each step of the scQDiff pipeline with real data and biological interpretations.

## 13.1 The Dataset

We will use a publicly available time-series scRNA-seq dataset of human CD4+ T-cells stimulated in vitro with anti-CD3/CD28 antibodies. The dataset includes samples from multiple time points: 0 hours (naive), 6 hours, 12 hours, 24 hours, 48 hours, and 72 hours post-stimulation. This provides a high-resolution view of the entire activation trajectory.

## 13.2 Step 1: Data Preprocessing and Pseudotime Inference

1.  **Quality Control**: Standard QC is performed to remove low-quality cells and genes.
2.  **Normalization and Feature Selection**: The data is normalized, and highly variable genes are selected for downstream analysis.
3.  **Dimensionality Reduction and Visualization**: The data is embedded in a low-dimensional space using UMAP for visualization. The UMAP plot shows a clear trajectory from the naive cells at 0 hours to the fully activated cells at 72 hours.
4.  **Pseudotime Inference**: Diffusion Pseudotime (DPT) is used to assign a continuous pseudotime value *t* ∈ [0, 1] to each cell, with *t* = 0 corresponding to the naive state.

## 13.3 Step 2: Learning the Drift Field

Since this dataset has splicing information, we use the **Velocity-Constrained Field Learning** method (Chapter 5.2). RNA velocity is computed for each cell, and a neural network is trained to learn the drift field *f*(*x*, *t*) that best predicts the velocity vectors. The network takes the gene expression vector and the pseudotime as input.

## 13.4 Step 3: Computing the Temporal Jacobian Tensor

Using the trained neural network for the drift field, we use the **Autograd Jacobian** method (Chapter 6.3) to compute the Jacobian matrix at a series of time points along the trajectory. This results in the temporal Jacobian tensor 𝓙, which captures the time-varying GRN of T-cell activation.

## 13.5 Step 4: Regulatory Archetype Extraction

We perform SVD on the unfolded Jacobian tensor to extract the regulatory archetypes and their temporal activations (Chapter 7). A scree plot suggests that *K* = 3 archetypes are sufficient to capture the majority of the regulatory dynamics.

-   **Archetype 1 (Early Response)**: This archetype is strongly active at early time points (0-12 hours). Gene set enrichment analysis shows it is enriched for genes involved in **NF-κB signaling**, **interferon response**, and **cytokine signaling** (e.g., *FOS*, *JUN*, *NFKB1*). This represents the initial, rapid response to T-cell receptor stimulation.

-   **Archetype 2 (Proliferation)**: This archetype peaks at intermediate time points (24-48 hours). It is highly enriched for **cell cycle** and **DNA replication** genes (e.g., *MKI67*, *PCNA*, *CDK1*). This corresponds to the massive clonal expansion phase of T-cell activation.

-   **Archetype 3 (Effector Differentiation)**: This archetype becomes dominant at late time points (48-72 hours). It is enriched for genes associated with **T-cell effector function**, such as *IFNG* (interferon-gamma), *IL2RA* (the high-affinity IL-2 receptor), and genes related to cytotoxicity. This represents the commitment to a specific effector lineage.

These three archetypes provide a beautiful, compact summary of the complex regulatory cascade of T-cell activation.

## 13.6 Step 5: Guided Trajectory Synthesis

We can now use these archetypes to perform predictive simulations.

**Forward Prediction**: Let's predict the effect of a hypothetical drug that inhibits the cell cycle. We can model this by applying a negative guidance profile to the Proliferation Archetype (Archetype 2). We simulate trajectories starting from naive T-cells with this control input. The results show that the simulated cells progress through the early activation phase but then stall and fail to expand, consistent with the effect of a cell cycle inhibitor.

**Reverse Engineering**: Can we design an intervention to make T-cells *more* effective? We can try to guide naive T-cells towards a "super-effector" state by applying a positive guidance profile to the Effector Differentiation Archetype (Archetype 3) at an earlier time point. Simulating this intervention suggests that it is possible to accelerate the differentiation process, leading to cells that express high levels of effector molecules like *IFNG* more rapidly.

## 13.7 Step 6: Calculating Control Energy

Finally, we can use control energy to quantify the difficulty of different interventions. We calculate the control energy required to drive a naive T-cell directly to the fully differentiated effector state in 24 hours, bypassing the normal 72-hour process. The resulting control energy is high, indicating that this forced, rapid differentiation is a difficult and likely inefficient process. In contrast, the energy required for the "super-effector" simulation (accelerating differentiation) is much lower, suggesting it is a more biologically plausible intervention.

This worked example demonstrates how scQDiff can be applied to a real dataset to extract meaningful biological insights. It moves beyond simple trajectory visualization to provide a mechanistic, predictive, and quantitative model of a complex cellular process. The next chapter will provide a similar worked example for a disease process: cancer progression and cancer progression.

# Chapter 14: Worked Example: Cancer Progression and Reprogramming

Having demonstrated scQDiff on a normal physiological process, we now turn to a disease context: cancer. Cancer progression is a form of aberrant cellular trajectory, where cells escape normal regulatory controls and evolve towards states of increased proliferation, invasion, and drug resistance. Understanding the dynamics of this process is critical for developing effective therapies. This chapter will walk through the application of scQDiff to a dataset of melanoma, illustrating how the framework can be used to model disease progression and design potential therapeutic strategies.

## 14.1 The Dataset

We will use a dataset of melanoma patient-derived xenografts (PDXs) that were treated with a BRAF inhibitor, a common targeted therapy for melanoma. The dataset includes samples taken before treatment (sensitive) and after the development of drug resistance (resistant). This provides us with two distinct cell populations, and our goal is to model the trajectory from the sensitive to the resistant state.

## 14.2 Step 1: Data Preprocessing and Trajectory Inference

1.  **Data Integration**: The sensitive and resistant cell populations are integrated and batch-corrected.
2.  **Trajectory Inference**: Since this is not a time-series dataset, we cannot use pseudotime directly. Instead, we use optimal transport-based trajectory inference (as implemented in tools like Waddington-OT) to construct a trajectory from the sensitive to the resistant cell populations. This defines the path along which we will model the dynamics.

## 14.3 Step 2: Learning the Drift Field

Given the discrete nature of the data (two endpoints), the **Bridge-Lite in Latent Space** method (Chapter 5.3) is the natural choice. We solve the Schrödinger Bridge problem between the sensitive and resistant cell distributions in a low-dimensional latent space. This yields the drift field *f*(*x*, *t*) that describes the dynamics of acquiring drug resistance.

## 14.4 Step 3: Computing the Jacobian and Extracting Archetypes

We compute the temporal Jacobian tensor from the learned drift field and decompose it into regulatory archetypes. Let's say we find two dominant archetypes:

-   **Archetype 1 (Proliferation/BRAF Signaling)**: This archetype is active in the sensitive cells and is characterized by genes downstream of the BRAF signaling pathway, as well as cell cycle genes. This represents the baseline oncogenic state driven by the BRAF mutation.

-   **Archetype 2 (Resistance Program)**: This archetype becomes active along the trajectory to the resistant state. It is enriched for genes involved in alternative signaling pathways (e.g., receptor tyrosine kinase signaling), drug efflux pumps, and epithelial-mesenchymal transition (EMT). This represents the adaptive rewiring of the GRN that allows cells to bypass the BRAF inhibitor.

## 14.5 Step 4: Designing Reprogramming Strategies

Our goal is to design a therapy that can either re-sensitize the resistant cells or prevent the sensitive cells from becoming resistant in the first place. We use **reverse engineering** (Chapter 8.4) to achieve this.

We simulate the reprogramming of resistant cells back towards the sensitive state by applying a control input based on the time-reversed endogenous dynamics. This involves applying a control that first counteracts the Proliferation/BRAF archetype and then strongly counteracts the Resistance Program archetype.

Analyzing the control input **u**(**x**, *t*) provides concrete therapeutic hypotheses. For example, if the control vector strongly pushes down on genes in the receptor tyrosine kinase pathway (part of the Resistance Archetype), it suggests that a combination therapy of a BRAF inhibitor and a receptor tyrosine kinase inhibitor could be effective in overcoming resistance.

## 14.6 Step 5: Quantifying Irreversibility with Control Energy

We calculate the control energy required to reprogram the resistant cells back to the sensitive state. Let's say we find that this energy is very high. This has a critical clinical implication: the acquisition of drug resistance in this model system is a highly stable, quasi-irreversible process. This suggests that once resistance is established, it may be very difficult to reverse with therapy. The more effective strategy might be to treat sensitive tumors with a combination therapy from the outset to *prevent* the emergence of the resistant state, a concept known as adaptive therapy.

We can also compare the control energy for different potential combination therapies. A therapy that requires lower control energy is predicted to be more efficient and effective. This allows for the in silico screening and ranking of different therapeutic strategies before they are tested in the lab.

This cancer-focused example highlights the utility of scQDiff in a disease context. It allows us to:

-   Model the dynamic regulatory changes that drive disease progression.
-   Deconstruct these changes into interpretable archetypes.
-   Use these archetypes to rationally design combination therapies.
-   Quantify the stability of the diseased state and the difficulty of reversing it.

This provides a powerful, data-driven engine for hypothesis generation in translational cancer research. The final chapters of the booklet will discuss how to experimentally validate these hypotheses and provide a guide to the practical implementation of the scQDiff software.

# Chapter 15: Experimental Validation Designs

A computational framework, no matter how mathematically elegant, is only as valuable as its ability to make accurate and testable predictions about the real biological world. The ultimate validation of scQDiff must come from experimental verification of its predictions. This chapter outlines three concrete experimental designs to rigorously test the core claims of the scQDiff framework: the accuracy of the learned Jacobian, the predictive power of forward synthesis, and the efficacy of the reverse-engineered reprogramming strategies.

## 15.1 Design 1: Perturbation-Based Validation of the Jacobian

**Core Claim**: The temporal Jacobian tensor, 𝓙(*t*), accurately represents the causal, time-varying gene regulatory network.

**Hypothesis**: Perturbing a key regulator identified by the Jacobian at a specific time will produce the predicted downstream gene expression changes.

**Experimental System**: A well-characterized, in vitro differentiation system, such as the differentiation of hematopoietic stem cells (HSCs) into erythrocytes and myeloid cells, or the differentiation of induced pluripotent stem cells (iPSCs) into neurons.

**Procedure**:

1.  **scQDiff Analysis**: Perform a time-course scRNA-seq experiment on the differentiation system and apply the full scQDiff pipeline to learn the temporal Jacobian tensor and regulatory archetypes.
2.  **Identify Key Regulators**: From the archetypes, identify key transcription factors (TFs) predicted to be major drivers at specific stages. For example, identify a TF (e.g., GATA1 for erythropoiesis) predicted to be a strong activator of lineage-specific genes in a late-stage archetype.
3.  **Time-Specific Perturbation**: Use a system for inducible perturbation, such as a doxycycline-inducible CRISPR interference (CRISPRi) or CRISPR activation (CRISPRa) system. Culture the cells and add doxycycline at the specific time point where the TF is predicted to be most active.
4.  **Readout**: Collect scRNA-seq data from the perturbed and control cell populations at a later time point.
5.  **Analysis**: Compare the observed gene expression changes in the perturbed cells to the predictions from scQDiff. The prediction is made by simulating the dynamics with a control input that mimics the CRISPR perturbation (e.g., a strong negative drift on the target TF). Success is defined as a high correlation between the predicted and observed changes in the downstream target genes of the perturbed TF.

## 15.2 Design 2: Prospective Validation of Guided Forward Synthesis

**Core Claim**: Guided forward synthesis can accurately predict the trajectory of cells under a novel perturbation.

**Hypothesis**: The population distribution of cells treated with a drug can be accurately predicted by simulating the drug's effect as a control input in scQDiff.

**Experimental System**: The T-cell activation system from Chapter 13.

**Procedure**:

1.  **scQDiff Training**: Use the original T-cell activation time-course data to train a full scQDiff model (drift field and archetypes).
2.  **Design In Silico Perturbation**: Choose a drug with a known mechanism of action that was *not* used in the training data. For example, a JAK inhibitor, which blocks cytokine signaling. Identify the archetype most closely associated with JAK-STAT signaling (likely the "Early Response" archetype).
3.  **Predictive Simulation**: Perform a guided forward synthesis simulation starting from the naive T-cell population, using a control input that applies a negative guidance to the JAK-STAT archetype. This generates a predicted scRNA-seq dataset for JAK inhibitor-treated T-cells at 72 hours.
4.  **Perform Experiment**: Culture naive T-cells and activate them in the presence of the JAK inhibitor. Collect scRNA-seq data at 72 hours.
5.  **Analysis**: Compare the real experimental dataset to the synthetic dataset generated by scQDiff. Success can be quantified in several ways:
    -   **Distributional Similarity**: Low Earth Mover's Distance (EMD) or Maximum Mean Discrepancy (MMD) between the real and synthetic cell distributions in gene space.
    -   **Marker Gene Expression**: High correlation in the expression levels of key marker genes for T-cell activation.
    -   **Classifier Accuracy**: Train a classifier to distinguish the real control vs. treated cells. If the classifier performs poorly when trying to distinguish the *synthetic* treated cells from the *real* treated cells, it indicates the simulation was highly realistic.

## 15.3 Design 3: Validation of Reverse Engineering for Reprogramming

**Core Claim**: Reverse engineering can identify effective, non-obvious combination therapies to overcome resistance or reprogram cell states.

**Hypothesis**: A combination therapy designed by scQDiff will be more effective at reversing a disease state than standard single-agent therapies.

**Experimental System**: The BRAF inhibitor-resistant melanoma model from Chapter 14.

**Procedure**:

1.  **scQDiff Analysis**: Analyze the sensitive vs. resistant melanoma cell data to identify the "Resistance Program" archetype.
2.  **Design Combination Therapy**: Use reverse engineering to design a control input that reverses the resistance trajectory. Analyze the components of this control vector to identify key genes or pathways that are being targeted. For example, the control might strongly push down on the expression of the AXL receptor tyrosine kinase.
3.  **Formulate Hypothesis**: The combination of the BRAF inhibitor and an AXL inhibitor will be more effective at killing resistant melanoma cells than either drug alone.
4.  **Perform Experiment**: Treat the resistant melanoma cell line with four conditions: (1) vehicle control, (2) BRAF inhibitor alone, (3) AXL inhibitor alone, and (4) the combination of both drugs. Measure cell viability (e.g., with a CellTiter-Glo assay) after 72 hours.
5.  **Analysis**: Test for a synergistic interaction between the two drugs. Synergy is confirmed if the effect of the combination therapy is significantly greater than the additive effect of the individual drugs. This would provide strong validation that scQDiff correctly identified a key vulnerability in the resistance program.

These three experimental designs provide a rigorous and multi-faceted approach to validating the core claims of the scQDiff framework. By moving from data-driven modeling to predictive simulation and experimental testing, we can close the loop between computation and biology, creating a powerful engine for discovery and therapeutic design.

# Chapter 16: Computational Implementation Guide

This final chapter provides a practical guide for researchers who wish to apply the scQDiff framework to their own data. While the preceding chapters have focused on the theoretical and mathematical underpinnings, this chapter will cover the practical aspects of implementation, including software requirements, data preparation, parameter selection, and interpretation of results. We will provide pseudocode for the main steps of the pipeline to illustrate the computational workflow.

## 16.1 Software Requirements

The scQDiff framework is implemented in Python and relies on several standard libraries for scientific computing and machine learning. A full implementation would typically require:

-   **Python** (version 3.8+)
-   **Scanpy**: For single-cell data preprocessing, analysis, and visualization.
-   **scVelo**: For RNA velocity computation.
-   **PyTorch** or **TensorFlow**: For building and training the neural networks for the drift field.
-   **NumPy** and **SciPy**: For numerical operations and scientific computing.
-   **POT (Python Optimal Transport)**: For solving the Schrödinger Bridge problem in the Bridge-Lite method.
-   **Matplotlib** and **Seaborn**: For plotting and visualization.

## 16.2 The scQDiff Pipeline: A Step-by-Step Workflow

Below is a high-level overview of the computational pipeline, from raw data to final insights.

### **Step 1: Data Preparation**

```python
# Pseudocode for data preparation
import scanpy as sc
import scvelo as scv

# Load data
anndata = sc.read_10x_mtx("path/to/data")

# Standard preprocessing
sc.pp.filter_cells(anndata, min_genes=200)
sc.pp.filter_genes(anndata, min_cells=3)
sc.pp.normalize_total(anndata, target_sum=1e4)
sc.pp.log1p(anndata)
sc.pp.highly_variable_genes(anndata, min_mean=0.0125, max_mean=3, min_disp=0.5)
anndata = anndata[:, anndata.var.highly_variable]

# Compute RNA velocity
scv.pp.moments(anndata, n_pcs=30, n_neighbors=30)
scv.tl.velocity(anndata)
scv.tl.velocity_graph(anndata)

# Infer pseudotime
scv.tl.latent_time(anndata)
```

### **Step 2: Learn the Drift Field**

This involves training a neural network. Here, we outline the velocity-constrained approach.

```python
# Pseudocode for drift field learning
import torch

# Define the neural network model for the drift field
class DriftNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # ... define layers (e.g., MLPs with residual connections)
    def forward(self, x, t):
        # ... forward pass
        return drift_vector

# Prepare data
x_data = torch.tensor(anndata.X)
t_data = torch.tensor(anndata.obs["latent_time"])
v_data = torch.tensor(anndata.layers["velocity"])

# Training loop
model = DriftNet(input_dim=x_data.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CosineSimilarity()

for epoch in range(num_epochs):
    optimizer.zero_grad()
    predicted_drift = model(x_data, t_data)
    loss = 1 - loss_fn(predicted_drift, v_data).mean()
    loss.backward()
    optimizer.step()
```

### **Step 3: Compute and Decompose the Jacobian**

```python
# Pseudocode for Jacobian computation and decomposition
import numpy as np

# Compute Jacobian at representative time points
time_points = np.linspace(0, 1, num_time_points)
jacobian_tensor = []
for t in time_points:
    # Get average cell state at time t
    x_avg = get_average_state(anndata, t)
    # Compute Jacobian using autograd
    jacobian_matrix = torch.autograd.functional.jacobian(
        lambda x: model(x, torch.tensor([t])), 
        x_avg
    )
    jacobian_tensor.append(jacobian_matrix.numpy())

# Unfold and decompose using SVD
J_unfold = np.array(jacobian_tensor).reshape(num_time_points, -1).T
U, S, Vt = np.linalg.svd(J_unfold, full_matrices=False)

# Extract archetypes and activations
num_archetypes = 3
archetypes = [U[:, k].reshape(G, G) for k in range(num_archetypes)]
activations = [S[k] * Vt[k, :] for k in range(num_archetypes)]
```

### **Step 4: Guided Synthesis and Control Energy**

```python
# Pseudocode for guided synthesis
def simulate_trajectory(x0, model, archetypes, guidance_profiles, gamma, num_steps):
    x = x0
    trajectory = [x0]
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i * dt
        endogenous_drift = model(x, torch.tensor([t]))
        
        # Calculate control input
        control_input = torch.zeros_like(x)
        for k, M_k in enumerate(archetypes):
            g_k_t = guidance_profiles[k](t)
            control_input += gamma * g_k_t * torch.matmul(torch.tensor(M_k), x)
        
        # Update state using Euler-Maruyama
        noise = torch.randn_like(x) * np.sqrt(2 * epsilon * dt)
        x = x + (endogenous_drift + control_input) * dt + noise
        trajectory.append(x)
    return trajectory
```

## 16.3 Parameter Selection and Best Practices

-   **Number of Genes**: The choice of highly variable genes is critical. Using too few may miss important regulators, while using too many increases computational cost and noise. A range of 2,000-5,000 genes is typical.
-   **Drift Network Architecture**: A simple MLP with 2-4 hidden layers and residual connections is often sufficient. The hidden layer size should be proportional to the input dimension.
-   **Number of Archetypes (*K*)**: As discussed in Chapter 7, use the scree plot of singular values as a primary guide. Start with a small *K* (e.g., 3-5) and increase it if the reconstruction error is high or if key biological processes appear to be missing.
-   **Guidance Strength (*γ*)**: This parameter controls the trade-off between following the endogenous dynamics and following the external guidance. A good starting point is to choose *γ* such that the magnitude of the control input is comparable to the magnitude of the endogenous drift.

## 16.4 Conclusion of the Booklet

We have journeyed from the abstract foundations of quantum mechanics to the practical implementation of a powerful computational framework for single-cell biology. The scQDiff framework provides a unified, principled, and interpretable approach to modeling cellular dynamics. By learning the time-varying regulatory network, decomposing it into fundamental archetypes, and using these archetypes to predict and control cellular behavior, scQDiff moves beyond descriptive analysis to offer a truly mechanistic understanding of how cells make decisions.

The path forward is exciting. The continued development of multi-omic and spatial technologies will provide ever-richer datasets for frameworks like scQDiff to consume. The integration of machine learning, optimal control theory, and experimental biology holds the promise of a new era of predictive and quantitative cell biology—an era where we can not only read the book of life but also begin to write its next chapters.

# Appendix A: Glossary of Terms

- **Archetype (Regulatory Archetype)**: A time-independent matrix representing a fundamental mode or program of gene-gene interactions. Extracted from the temporal Jacobian tensor via SVD.

- **Control Energy**: The total integrated squared magnitude of the control input required to drive a cell along a specific trajectory. A measure of the difficulty of a cellular transition.

- **Drift Field (*f*(*x*, *t*))**: A time-varying vector field that specifies the deterministic component of a cell's motion in gene expression space. It represents the average velocity of a cell at a given state and time.

- **Fokker-Planck Equation**: A partial differential equation that describes the time evolution of the probability density function of a population of particles undergoing a stochastic process.

- **Guided Trajectory Synthesis**: The process of simulating cellular trajectories under the influence of an external control input, used for forward prediction and reverse engineering.

- **Jacobian Matrix (𝓙(*t*))**: The matrix of all first-order partial derivatives of the drift field, where *J<sub>ij</sub>* quantifies the influence of gene *j* on the expression rate of gene *i* at time *t*.

- **Pseudotime**: A continuous variable that represents the progression of cells through a biological process, inferred from static single-cell data.

- **RNA Velocity**: The time derivative of the gene expression state, estimated from the ratio of unspliced to spliced mRNA transcripts.

- **Schrödinger Bridge Problem**: An optimal transport problem that seeks the most probable stochastic process connecting two given probability distributions at two different times.

- **Stochastic Differential Equation (SDE)**: A differential equation in which one or more terms is a stochastic process, used in scQDiff to model the trajectory of an individual cell.

- **Temporal Jacobian Tensor (𝓙)**: A three-dimensional tensor (G × G × T) formed by stacking the Jacobian matrices across all time points. It represents the complete, time-varying gene regulatory network.
'''# Appendix B: References

1.  **La Manno, G., Soldatov, R., Zehnder, A., et al.** (2018). RNA velocity of single cells. *Nature*, 560(7719), 494-498. [https://www.nature.com/articles/s41586-018-0414-6](https://www.nature.com/articles/s41586-018-0414-6)

2.  **Bergen, V., Lange, M., Peidli, S., Wolf, F. A., & Theis, F. J.** (2020). Generalizing RNA velocity to transient cell states through dynamical modeling. *Nature Biotechnology*, 38(12), 1408-1414. [https://www.nature.com/articles/s41587-020-0591-3](https://www.nature.com/articles/s41587-020-0591-3)

3.  **Schrödinger, E.** (1931). Über die Umkehrung der Naturgesetze. *Sitzungsberichte der Preussischen Akademie der Wissenschaften, Physikalisch-mathematische Klasse*, 144-153.

4.  **Aibar, S., González-Blas, C. B., Moerman, T., et al.** (2017). SCENIC: single-cell regulatory network inference and clustering. *Nature Methods*, 14(11), 1083-1086. [https://www.nature.com/articles/nmeth.4463](https://www.nature.com/articles/nmeth.4463)

5.  **Van de Sande, B., Flerin, C., Korcsmaros, T., et al.** (2020). A scalable SCENIC workflow for single-cell gene regulatory network analysis. *Nature Protocols*, 15(7), 2247-2276. (Describes SCENIC+)

6.  **Cuturi, M.** (2013). Sinkhorn distances: Lightspeed computation of optimal transport. *Advances in neural information processing systems*, 26. [https://papers.nips.cc/paper/2013/hash/af21d0c9783f7b7d3de4072b5f54639c-Abstract.html](https://papers.nips.cc/paper/2013/hash/af21d0c9783f7b7d3de4072b5f54639c-Abstract.html)

7.  **Street, K., Risso, D., Fletcher, R. B., et al.** (2018). Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics. *BMC genomics*, 19(1), 1-16.

8.  **Haghverdi, L., Büttner, M., Wolf, F. A., Buettner, F., & Theis, F. J.** (2016). Diffusion pseudotime robustly reconstructs lineage branching. *Nature methods*, 13(10), 845-848.
'''

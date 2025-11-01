# An Open Primer on Dynamic Gene Regulation and Quantum-Inspired Optimal Transport in Single-Cell Biology

This repository hosts a comprehensive, open-source primer on **scQDiff**, a conceptual and mathematical framework for modeling dynamic gene regulatory networks from single-cell data. This work bridges the fields of computational biology, quantum mechanics, and optimal control theory to provide a unified, interpretable, and predictive model of cellular decision-making.

## About This Project

The ability to understand and engineer the trajectories of cells is a central goal of modern biology. This primer introduces a novel approach inspired by the mathematics of quantum physics—specifically the Schrödinger Bridge problem—to infer the time-varying regulatory forces that govern cell state transitions. 

We move beyond static network inference and descriptive trajectory analysis to a framework that allows for:

-   **Mechanistic Insight**: Decomposing complex regulatory dynamics into a small number of interpretable "regulatory archetypes."
-   **Predictive Simulation**: Simulating the effect of novel perturbations and designing targeted interventions through guided trajectory synthesis.
-   **Quantitative Feasibility**: Calculating the "control energy" required to reprogram a cell from one state to another, providing a measure of biological irreversibility.

This repository is intended to be a living document and an educational resource for the scientific community.

## Repository Contents

This repository is organized into four main sections:

1.  **`/booklet`**: The full, 200+ page primer in Markdown format. This is a comprehensive, self-contained guide covering the entire framework, from the foundational theory to practical applications and implementation. It is organized into 16 chapters across four parts:
    -   **Part I: Foundations**: Introduces the biological problem, the quantum mechanics analogy, and the mathematical groundwork.
    -   **Part II: Core Framework**: Details the complete scQDiff pipeline, from learning the dynamics to extracting archetypes and calculating control energy.
    -   **Part III: Extensions**: Explores how the framework can be extended to multi-omic, spatial, and prior-knowledge-driven analyses.
    -   **Part IV: Applications and Validation**: Provides end-to-end worked examples and concrete experimental validation designs.

2.  **`/manuscripts`**: A pre-formatted, publication-ready manuscript suitable for submission to a computational biology journal or preprint server like bioRxiv. It presents the core conceptual advances of the scQDiff framework.

3.  **`/figures`**: A collection of all figures used in the booklet and manuscript, including conceptual diagrams of the temporal Jacobian tensor, multi-omic extensions, and the quantum-biology analogy.

4.  **`/code_examples`**: (Coming Soon) Jupyter notebooks and Python scripts providing practical implementations of the key components of the scQDiff pipeline.

## How to Use This Resource

-   **For a deep dive**, start with the `/booklet`. It is designed to be read sequentially and provides the most comprehensive treatment of the subject.
-   **For a high-level overview**, read the manuscript in the `/manuscripts` directory.
-   **For teaching or presentations**, feel free to use the diagrams in the `/figures` directory (with attribution).

## Citation

If you use the concepts, text, or figures from this work, please cite this GitHub repository. A formal citation will be added once the manuscript is available on a preprint server.

## Contributions

This is an open project, and we welcome contributions from the community. If you find errors, have suggestions for improvement, or would like to contribute new material (such as code examples or new applications), please open an issue or submit a pull request.

## License

This work is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). You are free to share and adapt this material for any purpose, even commercially, as long as you give appropriate credit.

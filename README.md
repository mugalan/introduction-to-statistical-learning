# introduction-to-statistical-learning
The aim of this repo is to introduce the essential elements of data science. Specifically, by providing an introduction to the ideas and techniques of data collection and management, summarizing and visualizing of data, statistical inference, and machine learning.

## Review of Probability Theory

This notebook provides a concise, measure-theoretic review of foundational probability concepts used throughout statistical learning. It combines formal definitions, intuitive explanations, simple numerical checks, and interactive visualizations (Plotly) to help bridge the gap between theory and practice.

### Overview
The notebook develops probability from first principles using the language of measure theory. It covers:
- Sample spaces and events
- σ-algebras and their role as "available information"
- Measures and measure-theoretic integration
- Probability measures, discrete (PMF) and continuous (PDF) formulations
- Key examples: counting measure, Lebesgue measure, Borel σ-algebra, and the Normal distribution
- Intuition-building examples and interactive visualizations (die partitions, unit disk partitions, coin PMF slider, Gaussian sliders)

### Contents (major sections)
1. Foundations of Probability Spaces
   - Sample space, events, set operations as logical operations
2. σ-algebras
   - Definition, motivation, examples (trivial σ-algebra, power set, Borel σ-algebra)
   - Role in representing information and measurability
3. Measure spaces and integration
   - Definition of a measure, countable additivity, integration w.r.t. a measure
   - Examples: counting measure, Lebesgue (volume) measure
4. Probability spaces
   - Definition of a probability measure P on (Ω, F)
   - PMF for discrete spaces and PDF (Radon–Nikodym derivative) for continuous spaces
5. Worked examples and visualizations
   - Bernoulli/coin PMF and interactive bar plot
   - Normal distribution with sliders for μ and σ
   - Partition examples on the unit disk (vertical vs horizontal) visualized with Plotly
   - Simple Python checks for σ-algebra closure and measurability

3## Key concepts explained
- Event as a subset of Ω; unions/intersections/complements map to logical OR/AND/NOT
- σ-algebra: closure under complement and countable unions; captures what is measurable
- Filtration idea (increasing σ-algebras) — relates to observing more information over time (Bayesian update intuition)
- Measure vs. probability measure: normalization P(Ω) = 1
- Absolute continuity and the Radon–Nikodym theorem: when a PDF exists relative to Lebesgue measure
- Distinctions between discrete and continuous probability representations (PMF vs PDF)
- Dimensionality and zero-measure intuition (points/lines/surfaces in higher-dimensional volume measure)

### Interactive code & visualizations
The notebook includes multiple interactive Plotly visualizations:
- Interactive PMF slider for coin toss (change p for Heads)
- Gaussian PDF with sliders for μ and σ
- Plotly scatter-based visualizations for partitions of the unit disk
These cells are designed to run inline (Jupyter/Colab) and illustrate the theoretical concepts visually.

### Why this notebook is useful for statistical learning
- Grounds common probability tools (PMF, PDF, measure, σ-algebra) in rigorous definitions used in theory-focused ML/statistics.
- Clarifies the differences between discrete and continuous modeling choices and why measure-theoretic language matters (e.g., point probabilities, zero-measure sets).
- Provides visual intuition and runnable examples that make abstract concepts accessible to practitioners and students.

### Google Colab notebook
- review_of_probability_theory.ipynb — [the full notebook](./review_of_probability_theory.ipynb) with narrative, proofs, examples, and visualizations.


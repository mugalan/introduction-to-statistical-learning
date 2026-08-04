# introduction-to-statistical-learning
The aim of this repo is to introduce the essential elements of data science. Specifically, by providing an introduction to the ideas and techniques of data collection and management, summarizing and visualizing of data, statistical inference, and machine learning.
## 1. Review of Probability Theory (`review_of_probability_theory.ipynb`)

- **Focus:** Rigorous, measure-theoretic foundations of probability theory and random variables.
- **Topics Covered:**
  - **Set Theory:** Set algebra, De Morgan's laws, partitions, and sample space decomposition.
  - **$\sigma$-algebras & Measurability:** Axioms, information granularity, Borel $\sigma$-algebras $\mathcal{B}(\mathbb{R})$, generated $\sigma$-algebras, and non-measurable set intuition.
  - **Measure & Integration:** Measure spaces $(\Omega, \mathcal{F}, \mu)$, counting measure, Lebesgue measure, and simple function Lebesgue integration.
  - **Probability Spaces:** Axioms of probability $(\Omega, \mathcal{F}, P)$, Discrete PMFs, and Bernoulli probability measures.
  - **Random Variables:** Measurable mappings, image measures, and distribution functions.
  - **Law of Large Numbers (WLLN) & Central Limit Theorem (CLT):** Establishes that the empirical mean of i.i.d. random variables converges in probability to the true expectation ($\bar{X}_n \xrightarrow{P} \mu$), while the distribution of their normalized sum converges in distribution to a standard Gaussian ($\mathcal{N}(0, 1)$).
- **Interactive Features:** Plotly 2D/3D interactive charts visualizing sample space partitions and Python code snippets for set operations and numerical simulations.

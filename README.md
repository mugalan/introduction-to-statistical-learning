# introduction-to-statistical-learning
The aim of this repo is to introduce the essential elements of data science. Specifically, by providing an introduction to the ideas and techniques of data collection and management, summarizing and visualizing of data, statistical inference, and machine learning.

---

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

### Key concepts explained
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

---

## Introduction to Information Entropy

This notebook surveys the many faces of entropy and information, tracing the concept from its origins in communication theory and thermodynamics to its central role in modern statistics and machine learning. The presentation emphasizes intuition, measure-theoretic foundations, and the differences between discrete and continuous settings.

### Historical context
- Claude Shannon (1948) formalized "information" as the reduction of uncertainty and introduced information entropy as a rigorous measure of uncertainty in a probability distribution.
- John von Neumann suggested naming Shannon’s measure “entropy” because of its formal similarity to thermodynamic entropy.
- Edwin T. Jaynes (1957) extended entropy into statistical inference: maximizing entropy subject to what is known yields the least-biased probability assignment consistent with available information.

### Core concepts (intuitive)
- Uncertainty: how much you do not know about an outcome before observing it.
- Self-information (surprisal): how much information a single observed outcome provides; rarer events carry more surprisal.
- Entropy: the average surprisal of a distribution — the expected amount of uncertainty (or information) in outcomes.
- Relative entropy / Kullback–Leibler (KL) divergence: a measure of how one probability distribution differs from another; it quantifies the information loss when using one distribution as an approximation of another and is always non-negative.
- Differential entropy: the continuous analogue of entropy defined relative to a reference measure (e.g., Lebesgue). It can be negative and is sensitive to coordinate scaling.
- Conditional entropy: the remaining uncertainty about one variable after observing another.
- Mutual information: the reduction in uncertainty about one variable due to observing another; symmetric and always non-negative.

### Measure-theoretic viewpoint (why the reference measure matters)
- Entropy is fundamentally a comparison of a probability measure to a reference measure. In discrete settings the counting measure is natural; in continuous settings the Lebesgue measure is typical.
- A distribution can have zero entropy relative to one reference measure and infinitely negative entropy relative to another. This resolves apparent paradoxes (e.g., point masses).

### Discrete vs. continuous differences (practical takeaways)
- Discrete entropy is always non-negative and interpretable as expected code length (bits or nats).
- Differential entropy can be negative and changes with scaling; it should be interpreted as a relative quantity (volume of uncertainty) rather than an absolute "number of bits."
- Gaussian distributions maximize differential entropy among continuous distributions with fixed variance. As a continuous distribution concentrates into a point (delta), differential entropy tends to negative infinity, while the corresponding discrete entropy (if treated as a point mass) is zero.

### Important properties and identities
- KL divergence is non-negative and equals zero only when the two distributions are identical (almost everywhere).
- Mutual information can be expressed as the difference between entropy and conditional entropy (information gained by observing another variable).
- Independence implies conditional entropy equals marginal entropy and mutual information is zero.

### Examples & visual intuition included in the notebook
- Visual comparisons showing lower vs. higher entropy distributions (concentrated vs. spread-out measures).
- Entropy of Gaussian distributions and the divergence of differential entropy as variance shrinks.
- A two-coin example that demonstrates computing entropy, conditional entropy, and mutual information for simple discrete variables.

### Practical notes for data science and ML
- Use KL divergence to assess model vs. true distribution differences (e.g., in variational inference).
- Mutual information is useful for feature selection and quantifying dependency between variables.
- Be careful when interpreting continuous entropy numerically — absolute values depend on units and scaling; consider relative measures (KL, mutual information) or discretization/quantization if you need bit-count interpretations.

### Google Colab notebook
- introduction_to_information_entropy.ipynb — [the full notebook](./introduction_to_information_entropy.ipynb).

---

## Multivariate Gaussian Distributions 

### Overview
- A multivariate Gaussian models a vector-valued random variable by specifying its center (mean) and how its components vary together (covariance). The covariance is typically described in blocks that separate one group of variables from another and their cross-dependencies.
- The joint density of a multivariate Gaussian is the familiar bell-shaped form determined entirely by the mean and covariance.

### Marginal distributions
- Any subset of components of a multivariate Gaussian is itself Gaussian. The marginal distribution for a subset uses the corresponding block of the full mean and the matching block of the full covariance.

### Conditional distributions
- Conditioning one subset of variables on observed values of another yields a Gaussian distribution for the conditioned subset.
- The conditional mean is an affine (linear + offset) function of the observed values, so observing the other variables shifts the mean predictably.
- The conditional covariance does not depend on the particular observed values; observing the other variables reduces uncertainty, so the conditional covariance is smaller (in the positive-semidefinite sense) than the marginal covariance.

### Law of total variance (matrix form)
- The overall (marginal) covariance of a subset decomposes into two parts: the expected conditional covariance (the residual uncertainty remaining after observing the other variables) plus the covariance of the conditional means (the part of variance explained by the other variables).
- Intuitively, this says total variation = unexplained variation + explained variation.

### Entropy and mutual information
- The differential entropy of a multivariate Gaussian is a simple function of its covariance; roughly speaking, larger covariance “volume” means larger entropy.
- Mutual information between two groups of variables measures how much observing one group reduces uncertainty about the other. For Gaussians, this mutual information depends only on the relevant covariance blocks and can be computed directly from them, reflecting the reduction in uncertainty when one set is observed.

### Practical note
- These closed-form marginal and conditional formulas are why multivariate Gaussians are used widely in linear regression, Kalman filtering, Gaussian process regression, and other estimation tasks where analytic updates and uncertainty quantification are valuable.

### Google Colab notebook
- Multivariate_Gaussian_Distributions.ipynb — [the full notebook](./Multivariate_Gaussian_Distributions.ipynb)
  
---

## Introduction to Conditional Expectation

### Overview
This notebook provides a rigorous, measure-theoretic introduction to conditional expectations, a foundational concept in probability theory, statistical learning, and Bayesian inference. It bridges the gap between abstract mathematical definitions and practical applications in machine learning.

### Key Sections

#### 1. **Definition of Conditional Expectation** 
- Introduces $\mathbb{E}[X|Y]$ as a unique random variable satisfying two properties:
  - **Measurability**: The conditional expectation is a function of $Y$ (via the Doob-Dynkin Lemma)
  - **Partial Averaging**: The integral property ensures that averages match on sets distinguishable by $Y$
- Grounded in the Radon-Nikodym Theorem from measure theory

#### 2. **Conditional Density ($f_{X|Y}$)**
- Derives the relationship between joint, marginal, and conditional distributions
- Shows how the conditional probability measure $P_{X|Y=y}$ is constructed
- Establishes the **Product Rule**:
  - Continuous case: $f_{X,Y}(x,y) = f_{X|Y}(x|y)f_Y(y)$
  - Discrete case: $p_{X,Y}(x,y) = p_{X|Y}(x|y)p_Y(y)$
- Introduces the **Disintegration Theorem**, which decomposes joint measures into conditional slices

#### 3. **Evaluating Conditional Expectation at $Y=y$**
- Clarifies the distinction between $\mathbb{E}[X|Y]$ (a random variable) and $\mathbb{E}[X|Y=y]$ (a numerical value)
- Shows that $\mathbb{E}[X|Y=y] = \int_{\mathscr{X}} x \, f_{X|Y}(x|y) \, dx$
- Explains the conditional random variable $X|Y=y$ as a family of distributions parameterized by $y$

#### 4. **Important Results**

##### **The Factorization Property**
- If $W$ is $\sigma(Y)$-measurable, then $\mathbb{E}[XW|Y] = W\mathbb{E}[X|Y]$
- Critical for manipulating conditional expectations when one term is already known

##### **Law of Total Expectation (Tower Property)**
$$\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]$$
- Enables solving complex problems by conditioning on an intermediate variable
- Includes Python verification demonstrating the principle with simulated data

##### **Law of Total Variance**
$$\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X|Y)] + \mathrm{Var}(\mathbb{E}[X|Y])$$
- **Unexplained Variance**: Remaining uncertainty after knowing $Y$
- **Explained Variance**: How much $Y$ accounts for $X$'s variability

##### **Optimality of Conditional Estimation**
- $\mathbb{E}[X|Y]$ is the unique minimizer of Mean Square Error (MSE)
- Establishes: $\mathbb{E}[(X - \mathbb{E}[X|Y])^2] = \mathbb{E}[\mathrm{Var}(X|Y)]$
- Any other estimator produces higher MSE

#### 5. **Applications to Machine Learning**

##### **Unified Multivariate Gaussian Framework**
For jointly normal distributions, the conditional expectation has a closed form:
$$\mathbb{E}[X|Y] = \mu_X + \Sigma_{XY}\Sigma_{YY}^{-1}(Y - \mu_Y)$$

This single formula unifies:
- Linear Regression (LR)
- Ordinary Least Squares (OLS)
- Kalman Filtering
- Gaussian Process Regression (GPR)

The conditional covariance provides the Mean Square Error:
$$\Sigma_{X|Y} = \Sigma_{XX} - \Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX}$$

#### **Bayesian Inference and MAP Estimation**
- $\mathbb{E}[X|Y]$ represents the Bayesian point estimate
- For Gaussian models, MAP estimation aligns with conditional expectation

##### **Dimensionality Reduction**
- Factor Analysis and Probabilistic PCA leverage conditional distribution properties to model high-dimensional data through low-dimensional latent variables

##### **Active Learning and Bayesian Optimization**
- Use conditional covariance to determine optimal sampling locations
- Prioritize regions of highest remaining uncertainty

##### **Information Theory**
- Conditional expectations underpin mutual information and information-theoretic measures in ML

### Learning Objectives
After studying this notebook, readers will understand:
1. The formal measure-theoretic definition of conditional expectations
2. How conditional distributions decompose joint distributions
3. Core theorems connecting to probability and statistics (Tower Property, Total Variance)
4. The optimality of $\mathbb{E}[X|Y]$ in estimation theory
5. Practical connections to Bayesian inference, linear models, and modern ML algorithms

### Prerequisites
- Familiarity with probability spaces and measure theory
- Understanding of random variables, distributions, and expectations
- Linear algebra (for multivariate applications)

### Google Colab notebook
- introduction_to_conditional_expectations.ipynb — [the full notebook](./introduction_to_conditional_expectations.ipynb).
This notebook serves as essential preparation for understanding state-of-the-art machine learning methods rooted in Bayesian inference and probabilistic modeling.

---

## Applications of Conditional Expectation

### Kalman Filter

#### Overview
The Kalman Filter is a recursive algorithm that computes the conditional expectation of a hidden state given noisy measurements. It elegantly demonstrates how conditional expectation concepts from probability theory translate into practical sequential estimation.

#### Key Concept
In the linear-Gaussian setting, the Kalman Filter updates the state estimate $\mathbb{E}[x_k|y_1, \ldots, y_k]$ through two steps:
- **Predict**: $m_k^- = A_{k-1} m_{k-1}$ (propagate expected state via system dynamics)
- **Update**: $m_k = m_k^- + K_k(y_k^{\text{obs}} - H_k m_k^-)$ (refine with measurement residual)

The Kalman Gain $K_k$ is an optimal, data-adaptive weighting that balances prediction uncertainty against measurement noise.

#### Examples Included
- **1-D Tracking**: Constant-velocity motion in one dimension
- **2-D Tracking**: Position tracking in the $x$-$y$ plane with noisy position measurements
- **HVAC Thermal Dynamics**: Tracking zone temperature, humidity, and CO₂ concentration from sensor data

#### Connection to Conditional Expectation
The notebook shows how:
- Joint Gaussian distributions enable closed-form conditional expectations
- The conditional covariance $P_k$ quantifies remaining uncertainty (Bayesian posterior)
- Error dynamics reveal how estimation error decays with better measurements

#### Google Colab notebook
- [Kalman_Filter.ipynb](./Kalman_Filter.ipynb) — interactive notebook with theory, derivations, simulations, and visualizations.

### Gaussian Process Regression

#### Overview 
Gaussian Process Regression (GPR) is a practical application of conditional expectation in a Bayesian framework. Given noisy observations of an underlying process, GPR computes the conditional expectation $\mathbb{E}[X_g \mid \mathscr{Y}_n = y_n]$ to produce both a predictive mean and uncertainty quantification. The approach treats the unknown function as a random variable, enabling probabilistic predictions that respect the correlation structure encoded by a kernel function.

#### Key Applications:
- **Denoising**: Filtering out measurement noise to recover the latent signal
- **Interpolation**: Inferring values in regions with missing data
- **Uncertainty Quantification**: Providing confidence intervals that widen in data-sparse regions

#### Google Colab notebook
- [gaussian_process_regression.ipynb](./gaussian_process_regression.ipynb) — interactive notebook with theory, derivations, simulations, and visualizations.

### Linear Regression

#### Overview 
A worked example showing how conditional expectation is used in multivariate linear regression is available in the repository. The notebook derives the conditional distribution Y | X = x (showing E[Y|X=x] = βx + β0 and Var[Y|X=x] = Σν), presents the maximum-likelihood / OLS solution, and includes a hands-on example with a green-building dataset.

#### Google Colab notebook
- [linear_regression.ipynb](./linear_regression.ipynb) — interactive notebook with theory, derivations, simulations, and visualizations.


### Bayesian Inference

A companion Jupyter notebook that gives a concise, measure-theoretic and practical introduction to Bayesian inference. It covers the mathematical foundations, the density (formula) form of Bayes' rule, Bayesian point estimation via conditional expectation, Gaussian conjugate examples, and several concrete examples (coin bias, loan default probability) with interactive visualizations.

#### Overview
This notebook treats Bayesian inference as "learning under uncertainty" and shows how a prior distribution over an unknown parameter X is updated to a posterior distribution after observing data Y. It emphasizes a measure-theoretic view (priors, likelihoods, and posteriors as probability kernels), shows the density form of Bayes' rule when densities exist, and demonstrates Bayesian point estimation (posterior mean as the conditional expectation and its optimality under squared error).

#### Topics covered (detailed)
- Motivation: Bayesian inference as rational belief revision; interpretation of parameters as unknown-but-fixed vs. uncertain.
- Probability-space formulation: X and Y as measurable maps on an underlying probability space; joint, marginal, and conditional laws.
- Regular conditional distributions and disintegration: posterior and likelihood as conditional probability kernels; the measure-theoretic Bayes identity.
- Density-level Bayes' rule: derivation of posterior density f(X|Y) ∝ f(Y|X) f_X(X) and normalization via marginal f_Y(Y).
- Bayesian point estimation: conditional expectation E[X | Y] as the posterior mean; uniqueness and optimality under mean-square error; conditional variance/covariance as remaining uncertainty.
- Closed-form Gaussian case: joint Gaussian X,Y and the linear update formula for the posterior mean and covariance (Kalman-like gain).
- Conjugate Beta–Binomial examples:
  - Coin bias: Beta(α, β) prior, Binomial likelihood → Beta(α+h, β+t) posterior; code to plot prior/posterior and means.
  - Loan default probability: domain-motivated Beta(5,95) prior updated by Binomial data (e.g., 16 defaults out of 200) to show posterior shift and interpretation.
- Interactive code: functions that plot prior and posterior Beta densities and annotate prior/posterior means (uses Plotly for interactivity).

#### What you will learn (learning outcomes)
- How to formalize Bayesian inference in measure-theoretic terms (priors, likelihoods, posterior kernels).
- How densities arise and how to apply Bayes' rule in practice.
- Why the posterior mean is a natural Bayesian point estimate and its optimality properties.
- How conjugate priors (Beta–Binomial, Gaussian–Gaussian) lead to closed-form updates.
- How to visualize prior-to-posterior updating and interpret shifts in beliefs.

#### Examples 
- Item Response Theory (IRT), which uses Bayesian hierarchical models to infer examinees' latent abilities and item parameters while properly accounting for uncertainty and pooling information across items/people,
- Click‑Through Rate (CTR) prediction, which uses Bayesian logistic or hierarchical models to produce probabilistic estimates of user click probabilities, enable online updating, and support uncertainty-aware decision making.

#### Google Colab notebook
- [Bayesian_Inference_Introduction.ipynb](./Bayesian_Inference_Introduction.ipynb) — interactive notebook with theory, derivations, simulations, and visualizations.

---

## Maximum Entropy Inference

This notebook provides a unified, pedagogical treatment of Jaynes' Principle of Maximum Entropy and its applications across physics and machine learning. Starting from the discrete Shannon entropy, it derives the generalized Boltzmann distribution and shows how the canonical, grand-canonical, and microcanonical ensembles arise as MaxEnt solutions. The continuous (Shannon–Jaynes) formulation and the Gaussian as a maximum-entropy distribution given mean and variance are presented. The notebook connects these ideas to machine learning by deriving the Maximum Entropy (conditional) classifier and showing how logistic regression / softmax arise from constrained entropy maximization. Worked examples include the binary entropy curve, the quantum harmonic oscillator, thermodynamic identities (partition function, free energies, heat capacity), and runnable Plotly code for illustrations.

### Overview
This repository contains a notebook that explains the Principle of Maximum Entropy (MaxEnt) and demonstrates its deep connections to statistical thermodynamics and machine learning. The content is both theoretical and practical: rigorous derivations are paired with numerical and visual examples.

### What’s included
- A clear statement of the Principle of Maximum Entropy and its justification as the least-biased inference given constraints.
- Derivation of the discrete MaxEnt solution and the generalized Boltzmann (exponential-family) distribution.
- Recovery of statistical mechanics ensembles:
  - Microcanonical ensemble → equal a priori probabilities and Boltzmann entropy S = kB ln Ω.
  - Canonical ensemble → partition function Z, temperature T = 1/(kB λ), Helmholtz free energy, and thermodynamic identities.
  - Grand canonical ensemble → chemical potential μ, grand potential, and Gibbs free energy connections.
- Thermodynamic relations and fluctuation formulae (e.g., energy variance ↔ heat capacity).
- Worked example: quantum harmonic oscillator — explicit Z(T), U(T), S(T), and CV(T).
- Continuous MaxEnt (Shannon–Jaynes) with a discussion of the reference measure m(x) and derivation of the Gaussian as the MaxEnt density given mean and variance.
- Machine learning application:
  - Conditional MaxEnt derivation for P(Y|X) and how logistic regression / softmax follows from MaxEnt constraints on feature expectations.
  - Practical notes on the equivalence between maximizing conditional entropy under feature constraints and training with cross-entropy.
- Interactive / runnable code snippets:
  - Plotly visualization of the binary entropy function.
  - Code scaffolding and examples suitable for Colab or a local Jupyter environment.

### Key equations and identities (high-level)
- Shannon entropy (discrete): H(p) = −∑ p_j ln p_j
- MaxEnt solution (discrete): p_j* = (1/Z) exp(−∑ λ_i f_i(x_j)), Z = ∑ exp(−∑ λ_i f_i(x_j))
- Continuous Shannon–Jaynes entropy: H_c[p] = −∫ p(x) ln(p(x)/m(x)) dx
- Canonical ensemble: p_j ∝ e^{−E_j/(k_B T)}, Z = ∑ e^{−E_j/(k_B T)}
- Thermodynamic relations: F = −k_B T ln Z, S = −(∂F/∂T)_V, U = F + TS
- Fluctuation relation: C_V = (1/(k_B T^2)) (⟨E^2⟩ − ⟨E⟩^2)
- Conditional MaxEnt → Softmax: P(Y=c | x) = exp(w_c · f(x)) / ∑_l exp(w_l · f(x))


### Learning outcomes / key takeaways
- MaxEnt provides a principled method for constructing least-biased probability models from partial information.
- Thermodynamic ensembles can be derived as MaxEnt solutions by choosing appropriate macroscopic constraints.
- The exponential-family structure of MaxEnt underpins many standard probability models and machine learning classifiers (e.g., softmax/logistic regression).
- Macroscopic thermodynamic quantities (entropy, free energy, heat capacity) follow from the partition function and can be interpreted in terms of microscopic fluctuations.

#### Google Colab notebook
- [maximum_entropy_inference_statistical_thermodynamics_and_ML.ipynb](./maximum_entropy_inference_statistical_thermodynamics_and_ML.ipynb) — interactive notebook with theory, derivations, simulations, and visualizations.

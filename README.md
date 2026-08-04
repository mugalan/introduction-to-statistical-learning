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

$#### **Bayesian Inference and MAP Estimation**
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

## Application of Conditional Expectation

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



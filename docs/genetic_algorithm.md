# Genetic algorithm for classification

**Fellipe Almassy** · November 2026

This document explains how a genetic algorithm can train a linear classifier. Instead of using gradients, the method evolves a population of candidate solutions: the best ones are more likely to reproduce, their offspring are mutated and replace the worst solutions. The text describes each step of the algorithm, with the mathematical notation and a numerical example, and points to the matching function of the `genetic-classifier` library.

## Goal

We start from a labeled dataset made of samples. Each sample has a feature vector and a known class. The goal of the genetic algorithm is to find the coefficients of a linear function that, from the features, tells the class of each sample. After training, this function can classify new data.

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of samples in the dataset |
| $n$ | number of features |
| $K$ | number of classes, numbered from $0$ to $K-1$ |
| $\mathbf{x} = (x_1, \dots, x_n)$ | features of a sample |
| $y$ | true class of a sample |
| $m$ | number of chromosomes in the population |
| $L = n + 1$ | number of genes in each chromosome |
| $h$ | number of offspring created per generation |
| $r$ | number of genes changed in each child |
| $s$ | number of chromosomes replaced per generation |

## Representing a solution

Each **chromosome** is a vector of real numbers that represents a candidate solution. It is made of a **bias** $b$ and one **gene** $g_j$ for each feature:

$$
\mathbf{c} = (b, g_1, g_2, \dots, g_n)
$$

With these values, the chromosome computes a score $q$ for any sample:

$$
q(\mathbf{x}) = b + g_1 x_1 + g_2 x_2 + \dots + g_n x_n = b + \sum_{j=1}^{n} g_j x_j
$$

Each gene is the weight of a feature in the decision. The bias shifts the score of every sample by the same amount. Without it, a sample with all features equal to zero would always get $q = 0$, and the decision boundary would be forced to pass through the origin.

In binary classification, the score is turned into a class by its sign:

$$
\hat{y} =
\begin{cases}
1, & \text{if } q(\mathbf{x}) \geq 0 \\
0, & \text{if } q(\mathbf{x}) < 0
\end{cases}
$$

In ordinal classification, the $K$ classes have an order (for example, bad < average < good). We define $K-1$ fixed cut points $t_1 < t_2 < \dots < t_{K-1}$, chosen by the user, which do not evolve during training. With the conventions $t_0 = -\infty$ and $t_K = +\infty$, the predicted class is:

$$
\hat{y} = k \quad \text{such that} \quad t_k \leq q(\mathbf{x}) < t_{k+1}
$$

## The algorithm loop

1. Create the initial population.
2. Compute the fitness of each chromosome.
3. Select a father and a mother by roulette.
4. Create $h$ offspring by crossover.
5. Mutate the offspring.
6. Replace the $s$ worst chromosomes with the $s$ best offspring and go back to step 2.

The loop ends when the best fitness found reaches the target value or when the maximum number of generations runs out. The result is the best chromosome seen during the whole training, kept aside, and not necessarily a chromosome of the last population.

### Step 1: creating the population

The initial population is a matrix $P$ with $m$ rows and $L$ columns. Each row is a chromosome and each column is a gene, the first column being the bias:

$$
P =
\begin{pmatrix}
b_1 & g_{1,1} & g_{1,2} & \cdots & g_{1,n} \\
b_2 & g_{2,1} & g_{2,2} & \cdots & g_{2,n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
b_m & g_{m,1} & g_{m,2} & \cdots & g_{m,n}
\end{pmatrix}
$$

All values are drawn uniformly from an interval, $[-1, 1]$ by default. The limits are parameters of the function and can be changed.

### Step 2: computing the fitness

The fitness is the grade of a chromosome: the better it classifies the samples of the dataset, the higher the grade. The most direct measure would be accuracy, that is, the fraction of correct answers. It is misleading, however, when the classes are imbalanced. In a dataset with 95 samples of class 1 and 5 of class 0, a chromosome that always answers 1 has 95% accuracy without having learned anything.

That is why we measure the recall of each class, that is, the fraction of samples of that class that the chromosome got right:

$$
R_k = \frac{\text{correct answers in class } k}{\text{total samples of class } k}
$$

The fitness is the product of the recalls:

$$
F = \prod_{k=0}^{K-1} R_k
$$

So $0 \leq F \leq 1$, and $F = 0$ whenever the chromosome gets every sample of some class wrong. In the case above, the chromosome that always answers 1 has $R_1 = 1$ and $R_0 = 0$, so $F = 0$. The result depends only on the fraction of correct answers within each class, not on how many samples each class has.

**Example.** Consider the dataset below and the chromosome $\mathbf{c} = (2, 3, -4, 1, -1)$. The values are integers and outside the default interval only to make the arithmetic easier. The score is:

$$
q(\mathbf{x}) = 2 + 3x_1 - 4x_2 + 1x_3 - 1x_4
$$

| | $x_1$ | $x_2$ | $x_3$ | $x_4$ | Class |
|---|---|---|---|---|---|
| Sample 1 | 1 | 0 | 1 | 1 | 1 |
| Sample 2 | 1 | 1 | 0 | 0 | 1 |
| Sample 3 | 1 | 0 | 0 | 1 | 0 |
| Sample 4 | 0 | 1 | 1 | 1 | 0 |

Applying the score to each sample and turning the result into a class with the binary rule:

| | Computation | $q$ | Predicted | True | Result |
|---|---|---|---|---|---|
| Sample 1 | $2 + 3 \cdot 1 - 4 \cdot 0 + 1 \cdot 1 - 1 \cdot 1$ | $5$ | 1 | 1 | correct |
| Sample 2 | $2 + 3 \cdot 1 - 4 \cdot 1 + 1 \cdot 0 - 1 \cdot 0$ | $1$ | 1 | 1 | correct |
| Sample 3 | $2 + 3 \cdot 1 - 4 \cdot 0 + 1 \cdot 0 - 1 \cdot 1$ | $4$ | 1 | 0 | wrong |
| Sample 4 | $2 + 3 \cdot 0 - 4 \cdot 1 + 1 \cdot 1 - 1 \cdot 1$ | $-2$ | 0 | 0 | correct |

The chromosome got 3 of the 4 samples right, an accuracy of 75%. Per class:

$$
R_1 = \frac{2}{2} = 1, \qquad R_0 = \frac{1}{2} = 0.5, \qquad F = R_1 \cdot R_0 = 0.5
$$

### Step 3: roulette selection

The fitnesses are normalized into probabilities that add up to 1:

$$
p_i = \frac{F_i}{\sum_{l=1}^{m} F_l}
$$

If every fitness is zero, every chromosome gets the same chance, $p_i = 1/m$.

To spin the roulette, we draw $u$ uniformly in $[0, 1)$ and pick the first chromosome whose cumulative probability exceeds $u$. The draw is done twice, once for the father and once for the mother, so both can be the same chromosome. A single couple is chosen per generation.

**Example.** With fitnesses 0.5, 0.25 and 0.25, the probabilities are the same, and the cumulative ones are 0.5, 0.75 and 1. If $u = 0.6$, the second chromosome is picked.

### Step 4: one-point crossover

For each of the $h$ offspring, we draw a cut point $c \in \{1, \dots, L-1\}$. The child gets the first $c$ genes from the father and the remaining genes from the mother:

$$
\mathbf{c}^{\text{child}} = (\mathbf{c}^{\text{father}}_{1}, \dots, \mathbf{c}^{\text{father}}_{c}, \mathbf{c}^{\text{mother}}_{c+1}, \dots, \mathbf{c}^{\text{mother}}_{L})
$$

Since the cut is never at the ends, the child always inherits at least one gene from each parent.

**Example.** With father $(2, 3, -4, 1, -1)$, mother $(-1, 0.5, 2, -3, 1)$ and $c = 2$, the child is $(2, 3, 2, -3, 1)$.

### Step 5: mutation

In each child, we draw $r$ different positions and replace the gene at each one with a new uniform value in $[-1, 1]$. Mutation brings into the population values that did not exist in the parents and therefore helps keep diversity. In the current implementation, this interval is fixed and does not follow the limits chosen in step 1.

**Example.** If the drawn position is the fourth gene and the new value is 0.4, the child $(2, 3, 2, -3, 1)$ becomes $(2, 3, 2, 0.4, 1)$.

### Step 6: updating the population

We compute the fitness of the mutated offspring. The $s$ chromosomes with the lowest fitness leave the population and the $s$ offspring with the highest fitness take their place, with $0 \leq s \leq \min(m, h)$. The rest of the population moves unchanged to the next generation, and the remaining offspring are discarded.

The swap does not compare children and parents: a child can enter even with a lower fitness than the chromosome it replaced. When $s < m$, the best chromosomes of the population are preserved, since only the worst ones leave.

## Parameters

Parameters of the `train` function and their default values:

| Symbol | Parameter | Default |
|---|---|---|
| $m$ | `n_chromosomes` | 30 |
| $r$ | `n_mutated_genes` | 2 |
| $h$ | `n_offspring` | 16 |
| $s$ | `n_replacements` | half of $h$ (8) |
| — | `n_generations` | 1000 |
| — | `target_fitness` | 1.0 |
| $t_1, \dots, t_{K-1}$ | `thresholds` | none (binary) |
| — | `gene_min`, `gene_max` | $-1$ and $1$ |

## Where each step is in the code

Each step is implemented in `src/genetic_classifier/`:

| Step | File | Function |
|---|---|---|
| 1. Creation | `creation.py` | `create_chromosomes` |
| 2. Fitness | `fitness.py` | `binary_fitness`, `ordinal_fitness` |
| 3. Selection | `fitness.py`, `selection.py` | `fitness_proportions`, `select_parents_roulette` |
| 4. Crossover | `crossover.py` | `one_point_crossover` |
| 5. Mutation | `mutation.py` | `mutate` |
| 6. Update | `update.py` | `update_population` |
| Full loop | `core.py` | `train` |
| Classification | `core.py` | `predict` |

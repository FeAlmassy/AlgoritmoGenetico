# Document: how the genetic algorithm works

Explanation of the algorithm implemented in this library, with mathematical notation and numerical examples.

| File | Content |
|---|---|
| `genetic_algorithm.md` | Markdown document, with formulas rendered by GitHub |
| `genetic_algorithm.tex` | Same content in LaTeX, to build a PDF |

## What the document covers

1. Goal of the algorithm and notation
2. How a solution is represented: chromosome, bias, genes and decision rules (binary and ordinal)
3. The six steps of the loop: creation, fitness, selection, crossover, mutation and update
4. Parameters of `train` and the function that implements each step

## How to build the PDF

With [Tectonic](https://tectonic-typesetting.github.io/), which downloads the required packages by itself:

```bash
sudo pacman -S tectonic        # on Arch; for other systems, see the website
cd docs
tectonic genetic_algorithm.tex
```

Or with a full LaTeX distribution, such as TeX Live:

```bash
cd docs
latexmk -pdf genetic_algorithm.tex
```

The PDF is written to `docs/genetic_algorithm.pdf`.

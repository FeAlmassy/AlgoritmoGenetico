# Documento: como funciona o algoritmo genético

Explicação do algoritmo implementado nesta biblioteca, com notação matemática e exemplos numéricos.

| Arquivo | Conteúdo |
|---|---|
| `algoritmo_genetico.md` | Documento em Markdown, com as fórmulas renderizadas pelo GitHub |
| `algoritmo_genetico.tex` | Mesmo conteúdo em LaTeX, para gerar PDF |

## O que o documento cobre

1. Objetivo do algoritmo e notação
2. Representação da solução: cromossomo, bias, genes e regras de decisão (binária e ordinal)
3. As seis etapas do ciclo: criação, fitness, seleção, cruzamento, mutação e atualização
4. Parâmetros de `treinar` e a função que implementa cada etapa

## Como gerar o PDF

Com o [Tectonic](https://tectonic-typesetting.github.io/), que baixa sozinho os pacotes necessários:

```bash
sudo pacman -S tectonic        # no Arch; em outros sistemas, veja o site
cd docs
tectonic algoritmo_genetico.tex
```

Ou com uma distribuição LaTeX completa, como o TeX Live:

```bash
cd docs
latexmk -pdf algoritmo_genetico.tex
```

O PDF sai em `docs/algoritmo_genetico.pdf`.

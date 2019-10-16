# Generating Labeled Genetic Synthetic Data to Non-Genetic Diseases

Creating large datasets for genetically influenced diseases is a hard and expensive task.
We intend to use the idea behind Generative Adversarial Networks to artificially create cohesive labeled genetic data to characterize genetically influenced diseases, specifically Dengue.

You can find the paper for the project [here](https://www.overleaf.com/project/5d8555ac21df820001d176ac).

## Getting Started

Just build and run the `csce` docker container:

```
docker build . -t csce
docker  run -it -v $PWD:/workspace csce bash
```

## Related Researches

- [Generative adversarial networks simulate gene expression and predict perturbations in single cells](https://www.biorxiv.org/content/10.1101/262501v2.full) - [Github](https://github.com/luslab/scRNAseq-WGAN-GP)

This work is already included in this repository in the directory `/src/scRNAseq-WGAN`. Unfortunately, the database used here is too large to a gitHub repository, therefore it has to be downloaded following the instructions on their [Github](https://github.com/luslab/scRNAseq-WGAN-GP).

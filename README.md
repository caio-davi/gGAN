# Generating Labeled Genetic Synthetic Data to Non-Genetic Diseases

Creating large datasets for genetically influenced diseases is a hard and expensive task.
We intend to use the idea behind Generative Adversarial Networks to artificially create cohesive labeled genetic data to characterize genetically influenced diseases, specifically Dengue.

> ##### Paper: [A Semi-Supervised Generative Adversarial Network for Prediction of Genetic Disease Outcomes](https://arxiv.org/abs/2007.01200). Guide to the paper implementation [here](./docker_version_readme.md). 

The branch `tamu_hprc` is meant to run on the [High Performance Research Computing (HPRC)](https://hprc.tamu.edu/) @Texas A&M University. Some environment configs are currently hard-coded on the code, so you'll need to make some changes to run it (I plan to move all of those to a new `.env` file as soon as possible). Also, this branch contains nightly modifications, which may be highly unstable at some points.


## Related Researches

- [Severe Dengue Prognosis Using Human Genome Data and Machine Learning](https://ieeexplore.ieee.org/abstract/document/8633395)
- [Characterization of a Dengue Patient Cohort in Recife, Brazil](https://www.ajtmh.org/content/journals/10.4269/ajtmh.2007.77.1128)
- [1000 Genomes project](https://www.nature.com/articles/nbt0308-256b)
- [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583)

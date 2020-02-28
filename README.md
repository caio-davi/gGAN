# Generating Labeled Genetic Synthetic Data to Non-Genetic Diseases

Creating large datasets for genetically influenced diseases is a hard and expensive task.
We intend to use the idea behind Generative Adversarial Networks to artificially create cohesive labeled genetic data to characterize genetically influenced diseases, specifically Dengue.

You can find the paper for the project [here](https://www.overleaf.com/project/5d8555ac21df820001d176ac).

## Getting Started

Just build and run the `ggan` docker container:

```
docker build . -t ggan
docker run -it -v $PWD:/gGAN ggan bash
```

After that, your environment should be set. Navigate to the src/ directory. You can run the gGAN using:

```
python gGAN.py 0.07
```
This will run the gGAN with the max allelic frequency threshold of 0.07, you can run `python gGAN.py -h` for help to know which are the available thresholds. These options are listed at the table in the next section, under the column "Frequency Proximity".
The number of epochs, the size of the batches, and all the network parameters are hardcoded into the python file, if you want to change that, you may edit the `gGAN.py` code. After running the gGAN model, the logs of the test will be available in the folder `run/`, you can plot your data using `plot_tests.py`, but it will need some hard code into the file as well.

## Models

We have diferent models for different samples sizes. 

| Model     | Frequency Proximity | #n SNPs | Sample Dimensions |
| --------- | ------------------- | ------- | ----------------- |
| gGan_3x4  | 0.07                | 12      | 3x4               |
| gGan_5x5  | 0.10                | 25      | 5x5               |
| gGan_8x12 | 0.21                | 96      | 8x12              |


## Related Researches

- [Severe Dengue Prognosis Using Human Genome Data and Machine Learning](https://ieeexplore.ieee.org/abstract/document/8633395)
- [Characterization of a Dengue Patient Cohort in Recife, Brazil](https://www.ajtmh.org/content/journals/10.4269/ajtmh.2007.77.1128)
- [1000 Genomes project](https://www.nature.com/articles/nbt0308-256b)
- [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583)

# Generating Labeled Genetic Synthetic Data to Non-Genetic Diseases

Creating large datasets for genetically influenced diseases is a hard and expensive task.
We intend to use the idea behind Generative Adversarial Networks to artificially create cohesive labeled genetic data to characterize genetically influenced diseases, specifically Dengue.


## Getting Started

Just build and run the `ggan` docker container:

```BASH 
docker build . -t ggan
docker run -it -v $PWD:/gGAN ggan bash
```

After that, your environment should be set. Navigate to the src/ directory. You can run the gGAN using:

```
python gGAN.py
```
Notice that is several parameters to run the Genetic GAN, bellow is the output of the help message:

```
$ python gGAN.py -h
usage: gGAN.py [-h] [--syn SYN] afd dim

positional arguments:
  afd         The threshold for the Allelic Freqeuncy Distance. Options are:
              0.07, 0.10, 0.21, SVM
  dim         Number of dimensions of the formated sample. Options are: 1
              (Conv1D) or 2 (Conv2D)

optional arguments:
  -h, --help  show this help message and exit
  --syn SYN   Run training+test with synthetic data.

```

The number of epochs, the size of the batches, and all the network parameters are hardcoded into the python file, if you want to change that, you may edit the `gGAN.py` code. After running the gGAN model, the logs of the test will be available in the folder `run/`, you can plot your data using `plot_tests.py`, but it will need some changes into the code as well.

## Related Researches

- [Severe Dengue Prognosis Using Human Genome Data and Machine Learning](https://ieeexplore.ieee.org/abstract/document/8633395)
- [Characterization of a Dengue Patient Cohort in Recife, Brazil](https://www.ajtmh.org/content/journals/10.4269/ajtmh.2007.77.1128)
- [1000 Genomes project](https://www.nature.com/articles/nbt0308-256b)
- [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583)

Here we describe all the steps to run the algorithms with the same configurations described in the [paper](https://arxiv.org/abs/2007.01200). You can find the source for this version on the [docker_version](./releases/tag/DockerVersion) tag.

## Getting Started

Just build and run the `ggan` docker container:

```
docker build . -t ggan
docker run -it -v $PWD:/gGAN ggan bash
```

After that, your environment should be set. Navigate to the src/ directory. You can run the gGAN using:

```
gGAN.py
```

Notice that are several parameters to run the Genetic GAN, bellow is the output of the help message:

```
gGAN --help
usage: gGAN [-h] [--afd AFD] [--syn SYN] cmd

positional arguments:
  cmd         Action to perform. Opitons: run : run, clear, test.

optional arguments:
  -h, --help  show this help message and exit
  --afd AFD   The threshold for the Allelic Freqeuncy Distance. Options are:
              0.07, 0.10, 0.21, SVM
  --syn SYN   Run training+test with synthetic data.
  
```

For now, the number of epochs, the size of the batches, and all the network parameters are hardcoded into the python file, if you want to change that, you may edit the `model.py` code. After running the gGAN model, the logs of the test will be available in the folder `run/`, you can plot your data using `plot_tests.py`, but it will need some changes into the code as well.
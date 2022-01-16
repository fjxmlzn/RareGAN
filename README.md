# RareGAN: Generating Samples for Rare Classes

**[[paper (AAAI 2022)](#)]**
**[[code](https://github.com/fjxmlzn/RareGAN)]**


**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Hao Liang ](https://www.linkedin.com/in/hao-liang-5b5020198/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/), [Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)

**Abstract:** We study the problem of learning generative adversarial networks (GANs) for a rare class of an unlabeled dataset subject to a labeling budget. This problem is motivated from practical applications in domains including security (e.g., synthesizing packets for DNS amplification attacks), systems and networking (e.g., synthesizing workloads that trigger high resource usage), and machine learning (e.g., generating images from a rare class). Existing approaches are unsuitable, either requiring fully-labeled datasets or sacrificing the fidelity of the rare class for that of the common classes. We propose RareGAN, a novel synthesis of three key ideas: (1) extending conditional GANs to use labelled and unlabelled data for better generalization; (2) an active learning approach that requests the most useful labels; and (3) a weighted loss function to favor learning the rare class. We show that RareGAN achieves a better fidelity-diversity tradeoff on the rare class than prior work across different applications, budgets, rare class fractions, GAN losses, and architectures.

---
This repo contains the codes for reproducing the experiments of our RareGAN in the paper. The codes were tested under Python 3.6.9, TensorFlow 1.15.2.

The code can be easily extended to your own applications, like [synthesizing images from rare classes](#extend-image), or [synthesizing data of more general formats (e.g., network packets, texts) for rare events (e.g., attacks)](#extend-network).

## Prerequisites

The codes are based on [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler) library, which helps you automatically schedule the jobs among GPU nodes. Please install it first. You may need to change GPU configurations according to the devices you have. The configurations are set in `config_generate_data.py` in each directory. Please refer to [GPUTaskScheduler's GitHub page](https://github.com/fjxmlzn/GPUTaskScheduler) for the details of how to make proper configurations.

## Image Experiments: Generating Rare Samples for CIFAR10 and MNIST

### CIFAR10

* Preparing the data according to the instructions [here](for_images/data/CIFAR10/README.md).
* Run

```
cd for_images
python -m scripts.CIFAR10.main_generate_data
```

### MNIST

* Preparing the data according to the instructions [here](for_images/data/MNIST/README.md).
* Run

```
cd for_images
python -m scripts.MNIST.main_generate_data
```

### <span id="extend-image">Your Own Image Dataset</span>
Simply add the data loading logic for your dataset [here](for_images/lib/data/load_data.py), and modify the training configuration file accordingly ([example](for_images/scripts/CIFAR10/config_generate_data.py)).

## System Experiments: Generating Network Packets for DNS Amplification Attacks and Packet Classifier Attacks

### DNS Amplification Attacks

* In [this configuration file](for_systems/scripts/DNS/config_generate_data.py), replace `<FILL IN IP ADDRESS>` with the IP address of the DNS server.
* Run

```
cd for_systems
python -m scripts.DNS.main_generate_data
```

> WARNING: During training, the code will generate a large number of DNS queries to the specified DNS server. Please make sure to use your own DNS servers in a sandboxed environment to avoid harming the public Internet. 

### Generating Packets that Trigger Long Processing Time for Packet Classifiers

To get an accurate evaluation of the packet processing time, we used separate servers for running RareGAN training and evaluating the processing time.

* On the server for evaluation, run

```
cd for_systems
python3 -m blackboxes.main_start_rpc_runner_server
```

* In [this configuration file](for_systems/scripts/PC/config_generate_data.py), replace `<FILL IN IP ADDRESS>` with the IP address of the evaluation server.
* Run

```
cd for_systems
python -m scripts.PC.main_generate_data
```

### <span id="extend-network">Your Own Dataset or Application</span>

The code supports a general data format and can be extended to any applications that want samples to have a large *metric* (e.g., packet amplification ratio in amplification attacks, or processing time of a system).

The following is all you need to do:

* A JSON configuration file that defines the data format. The format is defined as a list of [fields](for_systems/lib/data/fields.py). Examples: [DNS requests](for_systems/input_definitions/dns_input_definition.json), [network packets](for_systems/input_definitions/packet_classification_input_definition.json).
* Extend the [Blackbox class](for_systems/blackboxes/blackbox.py) and implements `query` interface that takes a list of samples as input, and returns their *metrics*. Examples: [packet size amplification ratio for DNS requests](for_systems/blackboxes/dns_blackbox.py), [packet classification time](for_systems/blackboxes/packet_classification_blackbox.py).
* Add your blackbox creation logic [here](for_systems/blackboxes/blackbox_utils.py). Note that there is a list of handy Blackbox wrappers that you can use (e.g., [off-loading the metric evaluation to a remote server](for_systems/blackboxes/rpc_wrapper.py), [evaluating each sample multiple times](for_systems/blackboxes/average_measurement_wrapper.py), [randomizing the order of samples](for_systems/blackboxes/random_order_measurement_wrapper.py), [warming up the system by evaluating random samples before the actual evaluation happens](for_systems/blackboxes/warm_up_measurement_wrapper.py). 
* Modify the training configuration file accordingly ([example](for_systems/scripts/DNS/config_generate_data.py)).



## Results

The code generates the following result files/folders:

* `<code folder>/results/<hyper-parameters>/worker.log`: Standard output and error from the code.
* `<code folder>/results/<hyper-parameters>/generated_data/data.npz`: Generated data from the rare class.
* `<code folder>/results/<hyper-parameters>/sample/*.png` (for image experiments only): Generated images during training.
* `<code folder>/results/<hyper-parameters>/checkpoint/*`: TensorFlow checkpoints and customized checkpoints.
* `<code folder>/results/<hyper-parameters>/time.txt`: Training iteration timestamps.
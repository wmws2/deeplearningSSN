<h2 align="center">Training Stochastic Stabilized Supralinear Networks by Dynamics-Neutral Growth</h2>

### 1. Overview
Code for the paper [Training Stochastic Stabilized Supralinear Networks by Dynamics-Neutral Growth](https://openreview.net/forum?id=znbTxnBPlx) as presented in NeurIPS 2022. The paper presents an introductory model where stochastic stabilized supralinear networks (SSNs) are trained to perform MNIST classification, followed by another model where SSNs are trained to perform sampling-based probabilistic inference under a Gaussian Scale Mixture (GSM). All models are built on Tensorflow 2.9.

<p align="center">
  <img src="figures/algoblock.png" width="600">
</p>

### 2. MNIST Classification Model
<p align="center">
  <img src="figures/mnist.png" width="600">
</p>

Code for the model can be found in <code>mnistmodel</code>. File descriptions are provided below.    <br>

**2.1 Code to be executed**  <br><br>
<code>1_autoencoder</code> compresses MNIST images in order to fit the number of excitatory neurons (set to 50 by default)  <br>
<code>2_initialization</code> randomly finds a stable network of 1 excitatory and 1 inhibitory neuron   <br>
<code>3_networkgrowth</code> trains the network to perform the classification task by dynamics-neutral growth  <br>
<code>4_fullnetwork</code> trains the built network for optimal performance  <br>
  <br>
**2.2 Other files**  <br><br>
<code>function_h</code> performs the non-linear input transformation into the SSN  <br>
<code>function_Tinv</code> computes the matrix of inverse time constants of the SSN  <br>
<code>main_settings</code> contains all relevant simulation and biological parameters of the model  <br>

### 3. Sampling-based Probabilistic Inference
<p align="center">
  <img src="figures/gsm.png" width="600">
</p>

Code for the model can be found in <code>samplingmodel</code>. Additionally, the [python version of CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) should be download and extracted to a folder called <code>./CIFAR-10/</code>. Some parameters are generated and stored in <code>./parameters/</code>.
File descriptions are provided below.    <br>

**3.1 Code to be executed**  <br><br>
<code>gsm_0_cifarprocessing</code> converts CIFAR-10 images into a single numpy binary file  <br>
<code>gsm_1_optimizefilters</code> optimizes for GSM filters by minimizing fraction of variance unexplained   <br>
<code>gsm_2_constructbank</code> converts optimized parameters into actual filters  <br>
<code>gsm_3_initialize</code> computes reasonable initial values of GSM parameters  <br>
<code>gsm_4_gsmtraining</code> optimizes GSM parameters for maximum likelihood of observing CIFAR-10 images  <br>
<code>gsm_5_computecov</code> pre-computing some matrices using an arbitrary-precision library  <br>
<code>gsm_6_inference</code> performs inference under GSM and computes posterior means and covariances  <br>
<code>gsm_7_networkinput</code> converts GSM computations into network inputs and targets  <br><br>
<code>ssn_1_initialization_targets</code> computes network targets for the small network <br>
<code>ssn_2_initialization</code> randomly finds a stable network of 1 excitatory and 1 inhibitory neuron   <br>
<code>ssn_3_initialization_ranking</code> ranks the best initializations with the lowest costs   <br>
<code>ssn_4_networkgrowth</code> trains the network to perform the inference task by dynamics-neutral growth  <br>
<code>ssn_5_fullnetwork</code> trains the built network for optimal performance  <br>

**3.2 Other files**  <br><br>
<code>function_h</code> performs the non-linear input transformation into the SSN  <br>
<code>function_Tinv</code> computes the matrix of inverse time constants of the SSN  <br>
<code>main_settings</code> contains all relevant simulation and biological parameters of the model  <br>

### 4. Citation

```
@inproceedings{soo2022-stablessn,
 author = {Soo, Wayne W.M. and Lengyel, M\'at\'e},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {tbd},
 pages = {tbd},
 publisher = {Curran Associates, Inc.},
 title = {Training Stochastic Stabilized Supralinear Networks by Dynamics-Neutral Growth},
 url = {tbd},
 volume = {35},
 year = {2022}
}
```

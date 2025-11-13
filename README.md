# Forward-Forward Conditional Convolutional Variational AutoEncoder

My implementation focuses on generating images using an architecture trained with the Forward-Forward Algorithm (FFA) instead of traditional backpropagation. At its core, the model is a modified version of the Conditional Variational Autoencoder (CVAE) built with convolutional neural networks (CNNs). Conditioning is done on the labels, which not only enables standard image reconstruction and latent space sampling—as in traditional VAEs—but also allows for class-conditional image generation.

The transition from backpropagation to FFA is achieved by introducing a modified version of the ELBO loss, computed locally at each decoder layer with respect to its symmetric encoder layer:

![ELBO equation](https://latex.codecogs.com/svg.image?\dpi{120}\mathcal{L}_{t}'(\boldsymbol{\theta},%20\boldsymbol{\phi};%20\mathbf{x}'^{(i)}_{t})%20\simeq%20\frac{1}{2}%20\sum_{j=1}^{J}%20\left(%201%20+%20\log((\sigma_j^{(i)})^2)%20-%20(\mu_j^{(i)})^2%20-%20(\sigma_j^{(i)})^2%20\right)%20+%20\frac{1}{L}%20\sum_{l=1}^{L}%20\log%20p_{\boldsymbol{\theta}}(\mathbf{x}'^{(i)}_{t}%20|%20\mathbf{z}^{(i,l)}))

 The encoder is trained layer by layer using the CwC (Class-wise Contrastive) loss:

![Loss Equation](https://latex.codecogs.com/svg.image?\dpi{120}\mathcal{L}_{t}=L_{CwC}%20=%20-\frac{1}{N}%20\sum_{n=1}^{N}%20\log\left(\frac{\exp(g_n^+)}{\sum_{j=1}^{J}%20\exp(G_{n,j})}\right))

This work is significant as it presents the first architecture that applies FFA for image generation using a CVAE, opening new directions for training deep generative models without backpropagation.

<img width="280" alt="Picture 1" src="https://github.com/user-attachments/assets/dd4210af-dd48-4ca7-9db4-a937dee44dec" />

## How to Use

- Install required dependencies from requirements.txt
- update config.py with the required parameters for choosing a dataset
- From the main folder, run the following command:
```bash
python main.py
```

# Latent Diffusion Model Architecture

Traditional diffusion models worked in high-dimensional pixel spaces, making them computationally expensive to train and incredibly slow to generate images. Because of this, they could not produce high-resolution outputs.

In other words, these traditional models processed the entire high-resolution image tensor as it was for most of the training. Imagine running heavy computations 1,000 times on a $1024 \times 1024 \times 3$ tensor. Only the GPU cost would dig holes in your pokets.

Latent Diffusion Models (LDMs) solved this by moving the heavy lifting into a smaller, compressed "latent" space. But before we deep-dive into the architecture that makes this possible, these are the basic mathematical terms we might come across:

- $t$: The time step. This usually goes from $0$ to $1$. We spilt our process into $T$ steps of size 1/$T$. 

- $z$: A specific, real image from our training dataset

- $p_{\text{data}}$: The true probability distribution of our data, where all real images in our dataset exist.

-  $p_{\text{init}}$: The initial distribution at $t=0$. This is usually pure, random noise (a Gaussian distribution in our case).

- $x$: A point in space at time $t$. This is the partially noisy data, between $p_{\text{init}}$ and $p_{\text{data}}$

- $p_t(x|z)$: The conditional probability path. If we start with a real image $z$, this is the distribution of our noisy image $x$ at time $t$ 

- $p_t(x)$: This is marginal probability path. If we look at all the noisy images at time $t$, this is their overall distribution, regardless of which original image they came from.

---
## Phase 1: 

Imagine a Latent Diffusion model as a car, and for a moment lets be mechanics trying to build the car. So heres the components we’re supposed to assemble.
### 1. The Variational Autoencoder (VAE)

The VAE is what separates the standard models like DDPMs from LDMs. It is made of two parts, as labeled $\mathcal{E}$ and $\mathcal{D}$ in the diagram:
- **The Encoder ($\mathcal{E}$):** This compresses raw, high-resolution images into a much smaller 2D latent space.
- **The Decoder ($\mathcal{D}$):** This learns to reconstruct compressed data back to a high-resolution image.
This drastically reduces computation costs. Instead of processing huge $1024\times1024\times3$ images, the main diffusion process happens in this small latent space. 
The VAE we use is trained independently on a diverse dataset of landscapes, faces, animals, cars so it learns to compress and decompress any concept, not just specific ones like faces.

### 2. The Discriminator & VGG Classifier

To make sure the VAE's Decoder is outputting realistic images, we train it with these two for parameter updation:

1.  **The PatchGAN Discriminator:** This is a Generative Adversarial Network trained simultaneously to look at tiny patches (like 70x70 pixels) of the decoded image, acting only as a texture critic. 
    Why texture? When uncertain, the safest option for L1 or L2 loss functions is to output the median or mean of all possible pixel values. Averaging distinct, sharp pixels (like dark and light green grass blade colours) creates blurry, smooth edges. Using only the Discriminator component, the PatchGAN adversarial loss penalises the blurriness this causes, forcing the Decoder to output only sharp, realistic pixels.

2.  **The VGG Image Classifier:** A pre-trained image classifier with frozen weights is used to calculate "Perceptual Loss." Instead of just comparing pixels directly, it compares the internal feature maps to ensure the generated image looks structurally identical to a real one to the human eye.

### 3. The CLIP Model

In an LDM, the text prompt conditions the image generation. But we can't input text directly into a neural networks; they understand the math in matrices, not English in alphabets.

This is where the **CLIP (Contrastive Language-Image Pre-training)** text encoder comes in. It converts human prompts like "a boy holding a coffee cup" into mathematical text embeddings of the same input size as of the noisy starting image inside the U-Net (discussed below). These embeddings get fed into the U-Net to guide the generation process.

### 4.  The U-Net

The U-Net puts the "diffusion" in the Latent Diffusion model. Its only motive is: learn how to remove added mathematical noise from the compressed latent representations.

It looks at the noisy latent vector, the timestep $t$ (how much noise was added), and the Text Prompt embeddings. It predicts what the noise lwas a step before and subtracts it to give a slightly cleaner input $x_{t-1}$. Its parameters ($\theta$) are updated using the Mean Squared Error (MSE) loss. 
 $$\text{Loss}_{LDM} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0,1), t} \left[ || \epsilon - \epsilon_\theta(z_t, t, \text{text}) ||^2 \right]$$
BONUS:
**Classifier-Free Guidance**: To force the model to follow the prompt, we randomly drop the text conditioning during training. During generation, the U-Net runs twice, once with the text and once without it and mathematically we push the generation in the direction of the conditioned prompt. 

---

## Phase 2: Inference
  
How do we put together our components the VAE, the U-Net, and the CLIP model to generate a concerningly realistic image? And from what? Here's what:

1. You type "a boy holding a coffee cup". The **CLIP text encoder** processes this prompt and turns it into a mathematical embedding.

2. The model samples a completely random tensor of pure noise _in the latent space_. This is our blank canvas, $p_{\text{init}}$ (only that it its not white, or a canvas) $$\text x_T \sim \mathcal{N}(0,1) $$3. This random noise is fed into the **U-Net**. Using **Cross-Attention**, the U-Net looks at the CLIP text embeddings and the input $z_{t}$ at timestep $t$ to understand what it is supposed make the final image look like. It uses Self-Attention to look at other image features across the whole canvas. Step-by-step, the U-Net predicts the noise in the latent space and outputs $z_{t-1}$.

3. After repeating this denoising loop $T$ times, we are left with a clean,  latent representation of a boy holding a coffee cup.

4. Finally, this clean latent tensor is passed through the **VAE Decoder**. The Decoder upscales and decompresses this mathematical representation back into the high-dimensional pixel space like it was trained to, leaving us with a high-resolution image of a boy holding a coffee cup.

Through all this all we try to do is to bring $p_{\text{init}}$ as close to $p_{\text{data}}$ as possible.

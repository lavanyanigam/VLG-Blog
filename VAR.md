# Visual Autoregressive Models (VAR):The Future?
![Some generated samples by VAR](images_blog/Generated_samples.png)

Autoregressive models have been quite successful in natural language processing (NLP) based tasks but whenever applied in the field of Computer Vision the results haven't been up to the mark. The main reason is that text has a natural sequential structure, whereas images are inherently two-dimensional and lack a canonical ordering. Flattening them into sequences disrupts their spatial structure.

That's where Visual Autoregressive Models (VARs) come into play. They address this issue by shifting from the classic approach of predicting the next token in a flattened sequence, they introduce a fundamentally different approach **next-scale prediction**. This shift allows autoregressive models to finally compete with, and even surpass, diffusion models in image generation.

So instead of treating an image as a long sequence VAR treats it like a heirarchy of resolutions or representations. It is very similar to the way humans draw, first a general structure and then refine the details.
## From Next Token to Next-Scale Prediction
Traditional autoregressive image models follow a fixed and simple pipeline:

1. Convert an image into discrete tokens using a tokenizer (e.g., VQ-VAE)
2. Flatten the 2D grid of tokens into a 1D sequence
3. Train a transformer to predict the next token given previous ones

But this algorithm has a lot of inefficiencies, flattening destroys spatial structure of images and generation because tokens are produced one at a time.

Now VARs replace this approach entirely. **Instead of modeling an image as a sequence of tokens, it models it as a sequence of resolutions**.

In traditional autoregressive models, the generation process looks like:

* token $\to$ token $\to$ token $\to$ $\dots$

In VARs it becomes:

* low-resolution image $\to$ higher resolution $\to$ even higher resolution $\to$ final image

 Concretely, an image is represented as a hierarchy of token maps:

* A very coarse representation (e.g., 1 $\times$ 1)
* Intermediate resolutions (e.g., 16 $\times$ 16)
* High-Resolution representation (e.g., 256 $\times$ 256)

Each level captures progressively finer details. The model then learns to generate the image coarse-to-fine, predicting each resolution conditioned on the previous ones.

So, it changes the autoregressive factorization from:

$$p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, x_2, \dots, x_{t-1})$$

to:

$$p(r_1, r_2, \dots, r_k) = \prod_{k=1}^{K} p(r_k \mid r_1, r_2, \dots, r_{k-1})$$

where each $r_k$ isn't just a single token, but an entire grid of tokens at resolution $h_k \times w_k$

## Multi-Scale Tokenization
To implement this idea, VARs rely on a multi-scale quantization autoencoder. This autoencoder converts an image into multiple discrete representations at different resolutions.

**Multi-Scale Encoding(Algorithm 1)**

The encoding process converts an image into its multi-scale token representation.

It works as follows:

1. Encode the image

    The input image is first passed through an encoder to produce a feature map:
   
   $$[f = \mathcal{E}(\text{im})]$$

2. Iterate over scales
    For each resolution level k = 1 to K:
    * The feature map is resized (interpolated) to the target resolution $h_k \times w_k$
    * This resized feature map is then quantized and stored as $r_k$:
      
$$[
r_k = \mathcal{Q}(\text{interpolate}(f, h_k, w_k))
]$$


3. Reconstruct approximation at this scale
The tokens are mapped back to feature space using the shared codebook:

$$[
z_k = \text{lookup}(Z, r_k)
]$$ 

  - This is then upsampled back to the original resolution

4. Residual refinement

    A key step: the model subtracts this reconstructed signal from the feature map:
   
$$[
f = f - \phi_k(z_k)
]$$

Here, $\phi_k$ is a small convolutional module that processes the reconstructed features.

This multi-scale representation aligns closely with how humans perceive images. We first grasp the global structure (shapes, layout), and only then focus on fine details (textures, edges). VAR explicitly models this hierarchy.

## Generation and Training Mechanism
Once the image is represented as a sequence of token map, the transformer is trained to generate these images sequentially across scales. In the generation process the transformer first generates lower resolution tokens and progressively generates higher resolution tokens.

 A key advantage of this approach is that all tokens within a given scale are generated in parallel. Unlike traditional autoregressive models that generate one token at a time, VAR generates an entire grid at once for each scale. This dramatically reduces inference time.

To maintain the autoregressive property, VAR uses a block-wise causal attention mask during training. This ensures that when predicting a token map at scale k, the model only has access to token maps from previous scales (1 to k−1), and not future ones, 
preserving the autoregressive property and preventing information leakage.

Within a scale, however, tokens are generated simultaneously and can attend to each other freely. This is a major departure from standard autoregressive models, where each token can only attend to previous tokens.

During inference, the process becomes even more efficient. Since generation proceeds strictly from lower to higher resolutions, there is no need for masking. Additionally, key-value (KV) caching can be ;
used to reuse computations from previous steps, further speeding up generation.

**Multi-Scale Reconstruction(Algorithm 2)**

1. **Initialize an empty feature map**
   
$$\hat{f}=0$$

2. **Iterate over scales**

    For each scale k = 1 to K :

   - Retrieve tokens $r_k$

   - Convert them back to feature vectors:
      
     $$z_k = \text{lookup}(Z, r_k)$$

   - Upsample to full resolution

   - Add the contribution to the feature map:

     $$\hat{f} = \hat{f} + \phi_k(z_k)$$

3. **Decode the final feature map**
   
   $$\hat{\text{im}} = \mathcal{D}(\hat{f})$$

So after the token maps have been generated by transformers across multiple scales then the decoder network reconstructs the final image via algorithm 2.


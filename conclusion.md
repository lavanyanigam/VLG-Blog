# Conclusion: Diffusion vs. Autoregression
AI image generation is evolving at an extreme pace. For a long time, Latent Diffusion Models (LDMs) were the undisputed choice for visual tasks. By shifting the computational heavy compute to a compressed latent space, LDMs enabled high-quality and conditioned image generation. However, Visual Autoregressive Models (VAR) now provide a cut-throat competition. By abandoning the "next-token" flat sequence to a "next-scale" hierarchy, VARs solve the spatial and computational issues that earlier held autoregressive vision models back.

So, back to our initial question: when to use diffusion, and when does autoregression make more sense?

**Choose Latent Diffusion Models (LDMs) for Control**: 
If you want precise text conditioning, fine-grained image editing, inpainting, or outpainting, LDMs are your best option. Their iterative denoising process is built for guided adjustments and highly specific user prompts.

**Choose Visual Autoregressive Models (VAR) for Speed and Scaling**: 
If you want fast inference speed, amazing raw image quality, and or massive datasets to train on, VAR is the clear choice. Because they benefit from the same predictable scaling laws that Large Language Models (LLMs) use, more data and compute gives better results for VAR models.

But ultimately, we are not looking at a "winner" in AI image generation. While LDMs currently dominate the practical, consumer focused side of AI art, VARs have proven that the autoregressive approach is far from obsolete in computer vision. Maybe, the next massive breakthrough in generative AI might just be a fusion of both, combining the predictable, fast scaling of autoregression with the pixel-perfect controllability of diffusion.

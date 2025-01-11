# Peekaboo: Text to Image Diffusion Models Are Zero-Shot Segmentors
#
# Copyright (c) 2023 Ryan Burgert
#
# This code is based on the Stable-Dreamfusion codebase's 'sd.py' by Jiaxiang Tang (https://github.com/ashawkey/stable-dreamfusion)
# which is licensed under the Apache License Version 2.0.
# It has been heavily modified to suit Peekaboo's needs, but the basic concepts remain the same.
# Tensor shape assertions have been added to the code to make it easier to read.
#
# Author: Ryan Burgert

#TODO: Use loss, add type annotations to add/remove noise, add denoised image func for ddpm attempt

from typing import Union,List,Optional

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor, logging
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import rp

# Suppress partial model loading warning
logging.set_verbosity_error()

_stable_diffusion_singleton = None #This singleton gets set the first time a StableDiffusion is constructed. Usually you'll only ever make one.

def _get_stable_diffusion_singleton():
    if _stable_diffusion_singleton is None:
        assert False, 'Please create a stable_diffusion.StableDiffusion instance before creating a label'
    return _stable_diffusion_singleton


class StableDiffusion(nn.Module):
    def __init__(self, device='cuda', checkpoint_path="sd/models--CompVis--stable-diffusion-v1-4", pipe=None):
        # Add hybrid image parameters
        self.hybrid_blend_ratio = 0.5  # Default blend ratio for hybrid images
        self.mask_blur_radius = 5  # Gaussian blur radius for mask edges
        self.mask = None  # Initialize mask as None
        
        global _stable_diffusion_singleton
        if _stable_diffusion_singleton is not None:
            rp.fansi_print('WARNING! StableDiffusion was instantiated twice!','yellow','bold')
        #Set the singleton. Other classes such as Label need this.
        _stable_diffusion_singleton=self
            
        super().__init__()

        self.device = torch.device(device)
        self.num_train_timesteps = 1000
        
        # Timestep ~ U(0.02, 0.98) to avoid very high/low noise levels
        self.min_step = int(self.num_train_timesteps * 0.02) # aka 20
        self.max_step = int(self.num_train_timesteps * 0.98) # aka 980

        print('[INFO] sd.py: loading stable diffusion...please make sure you have run `huggingface-cli login`.')
        
        # Unlike the original code, I'll load these from the pipeline. This lets us use dreambooth models.
        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float,
                requires_safety_checker=False,
                safety_checker=None,
            )
        
        pipe.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps) #Error from scheduling_lms_discrete.py
        
        self.pipe         = pipe
        self.vae          = pipe.vae         .to(self.device) ; assert isinstance(self.vae          , AutoencoderKL       ),type(self.vae          )
        self.tokenizer    = pipe.tokenizer                    ; assert isinstance(self.tokenizer    , CLIPTokenizer       ),type(self.tokenizer    )
        self.text_encoder = pipe.text_encoder.to(self.device) ; assert isinstance(self.text_encoder , CLIPTextModel       ),type(self.text_encoder )
        self.unet         = pipe.unet        .to(self.device) ; assert isinstance(self.unet         , UNet2DConditionModel),type(self.unet         )
        self.scheduler    = pipe.scheduler                    ; #assert isinstance(self.scheduler    , PNDMScheduler       ),type(self.scheduler    )
        
        self.uncond_text=''

        # Initialize CLIP image encoder
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.checkpoint_path=checkpoint_path
            
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] sd.py: loaded stable diffusion!')

    def get_image_features(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Extract detailed visual features from input images"""
        if isinstance(images, Image.Image):
            images = [images]
            
        # Process images through CLIP encoder
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Get all hidden states to capture more visual details
            outputs = self.image_encoder(**inputs, output_hidden_states=True)
            # Use both CLS token and patch embeddings
            features = torch.cat([
                outputs.last_hidden_state[:, :1, :],  # CLS token
                outputs.hidden_states[-2][:, 1:, :].mean(dim=1, keepdim=True)  # Average patch embeddings
            ], dim=1)
            
        return features

    def get_text_embeddings(self, prompts: Union[str, List[str]], image_features: Optional[torch.Tensor] = None)->torch.Tensor:
        """Get text embeddings with optional image feature conditioning"""
        if isinstance(prompts,str):
            prompts=[prompts]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompts, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt').input_ids

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.to(self.device))[0]
            
            # Condition text embeddings with image features if provided
            if image_features is not None:
                # Project image features to text embedding space
                image_features = image_features.expand(text_embeddings.shape[0], -1, -1)
                text_embeddings = text_embeddings + 0.5 * image_features  # Blend text and image features

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([self.uncond_text] * len(prompts), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt').input_ids

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]

        assert len(uncond_embeddings)==len(text_embeddings)==len(prompts)==len(text_input)==len(uncond_input)

        output_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        assert (uncond_embeddings==torch.stack([uncond_embeddings[0]]*len(uncond_embeddings))).all()
        assert (uncond_embeddings==uncond_embeddings[0][None]).all()

        assert output_embeddings.shape == (len(prompts)*2, 77, 768)

        return output_embeddings

    def add_noise(self, original_samples, noise, timesteps):
        #This is identical to scheduler.add_noise, assuming the scheduler is DDIM, DDPM or PNDM
        #It was copy-pasted
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.scheduler.alphas_cumprod[timesteps] ** 0.5
        # sqrt_alpha_prod = self.scheduler.match_shape(sqrt_alpha_prod, original_samples)
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps].to(self.device)) ** 0.5
        # sqrt_one_minus_alpha_prod = self.scheduler.match_shape(sqrt_one_minus_alpha_prod, original_samples)

        noisy_latents = sqrt_alpha_prod.to(self.device) * original_samples.to(self.device) + sqrt_one_minus_alpha_prod * noise
        return noisy_latents

    def remove_noise(self, noisy_latents, noise, timesteps):
        #TODO: Add shape assertions
        #This is the inverse of add_noise
        timesteps = timesteps.cpu()
        sqrt_alpha_prod = self.scheduler.alphas_cumprod[timesteps] ** 0.5
        # sqrt_alpha_prod = self.scheduler.match_shape(sqrt_alpha_prod, noisy_latents)
        sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[timesteps].to(self.device)) ** 0.5
        # sqrt_one_minus_alpha_prod = self.scheduler.match_shape(sqrt_one_minus_alpha_prod, noisy_latents)

        original_samples = (noisy_latents - sqrt_one_minus_alpha_prod * noise) / sqrt_alpha_prod.to(self.device)
        return original_samples
    
    def predict_noise(self, noisy_latents, text_embeddings, timestep):
        return self.unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings)['sample']

    def train_step(self,
                   text_embeddings: torch.Tensor,
                   pred_rgb: torch.Tensor,
                   guidance_scale: float = 100,
                   t: Optional[int] = None,
                   noise_coef: float = 1,
                   latent_coef: float = 0,
                   image_coef: float = 0,
                   mask: Optional[torch.Tensor] = None,
                   target_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Training step with enhanced mask and frequency handling
        
        Args:
            text_embeddings: Text embeddings for conditioning
            pred_rgb: Predicted RGB image
            guidance_scale: Scale for classifier-free guidance
            t: Timestep
            noise_coef: Coefficient for noise prediction loss
            latent_coef: Coefficient for latent prediction loss
            image_coef: Coefficient for image prediction loss
            mask: Optional mask for selective blending
            target_image: Optional target image for perceptual losses
        """
        # Loss weights
        ssim_weight = 0.4
        low_freq_weight = 0.3
        high_freq_weight = 0.2
        mask_consistency_weight = 0.1
        
        # Initialize losses
        losses = {}
        
        # Calculate SSIM loss if target image provided
        if target_image is not None:
            from source.msssim import MSSSIM
            msssim = MSSSIM()
            losses['ssim'] = 1 - msssim(pred_rgb_512.unsqueeze(0), target_image.unsqueeze(0))
            
            # Frequency separation loss
            pred_low = F.avg_pool2d(pred_rgb_512, kernel_size=4)
            target_low = F.avg_pool2d(target_image, kernel_size=4)
            pred_high = pred_rgb_512 - F.interpolate(pred_low, scale_factor=4, mode='bilinear')
            target_high = target_image - F.interpolate(target_low, scale_factor=4, mode='bilinear')
            
            losses['low_freq'] = F.mse_loss(pred_low, target_low)
            losses['high_freq'] = F.mse_loss(pred_high, target_high)
            
            # Mask consistency loss
            if mask is not None:
                masked_pred = pred_rgb_512 * mask
                masked_target = target_image * mask
                losses['mask_consistency'] = F.mse_loss(masked_pred, masked_target)
            
            # Apply loss weights
            total_loss = (ssim_weight * losses['ssim'] +
                         low_freq_weight * losses['low_freq'] +
                         high_freq_weight * losses['high_freq'] +
                         mask_consistency_weight * losses.get('mask_consistency', 0))
            
            # Backpropagate total loss
            pred_rgb_512.backward(gradient=w * total_loss, retain_graph=True)
        
        # This method is responsible for generating the dream-loss gradients.
        
        # interp to 512x512 to be fed into vae
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        assert 0<=t<self.num_train_timesteps, 'invalid timestep t=%i'%t

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512).to(self.device)

        
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents).to(self.device)
            #This is the only place we use the scheduler...the add_noise function. What's more...it's totally generic! The scheduler doesn't impact the implementation of train_step...
            latents_noisy = self.add_noise(latents, noise, t) #The add_noise function is identical for PNDM, DDIM, and DDPM schedulers in the diffusers library
            #TODO: Expand this add_noise function, and put it in this class. That way we don't need the scheduler...and we can also add an inverse function, which is what I need for previews...that subtracts noise...
            #Also, create a dream-loss-based image gen example notebook...

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.predict_noise(latent_model_input, text_embeddings, t)

            latent_pred = self.remove_noise(latents_noisy, noise_pred, t)
            output = latent_pred
            
            if image_coef:
                image_pred = self.decode_latents(latent_pred)

                
        #TODO: Different guidance scales for each type...if mixing them is useful...
                
        w = (1 - self.alphas[t])
            
        # perform noise guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        noise_delta=noise_pred - noise
        total_delta=noise_delta * noise_coef
        
        # Ryan's Latent Guidance
        latent_pred_uncond, latent_pred_text = latent_pred.chunk(2)
        latent_pred = latent_pred_uncond + guidance_scale * (latent_pred_text - latent_pred_uncond)
        latent_delta=latent_pred - latents
        total_delta=total_delta + latent_delta * latent_coef
        
        output=torch.stack([*output, *latent_pred])
        
        if image_coef:
            # Ryan's Image Guidance
            image_pred_uncond, image_pred_text = image_pred.chunk(2)
            image_pred = image_pred_uncond + guidance_scale * (image_pred_text - image_pred_uncond)
            image_delta=image_pred - pred_rgb_512
            pred_rgb_512.backward(gradient = w * image_delta * image_coef, retain_graph=True)


        # w(t), sigma_t^2
        grad = w * total_delta

        # manually backward, since we omitted an item in grad and cannot simply autodiff
        latents.backward(gradient=grad, retain_graph=True)

        return output

    def produce_latents(self, text_embeddings:torch.Tensor, height:int=512, width:int=512, num_inference_steps=50, guidance_scale=7.5, latents=None)->torch.Tensor:
        assert len(text_embeddings.shape)==3 and text_embeddings.shape[-2:]==(77,768)
        assert not len(text_embeddings)%2
        num_prompts = len(text_embeddings)//2

        if latents is None:
            latents = torch.randn((num_prompts, self.unet.in_channels, height // 8, width // 8), device=self.device)

        assert 0 <= num_inference_steps <= 1000, 'Stable diffusion appears to be trained with 1000 timesteps'

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                assert int(t) == t and 0 <= t <= 999, 'Suprisingly to me...the timesteps were encoded as integers lol (np.int64)'
                assert int(i) == i and 0 <= i <= 999, 'And because there are 1000 of them, the index is also bounded'
                t=int(t) # This akes some schedulers happy; it's the same value anyway.

                # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2) #The first half is the blank prompts (repeated); the second half is 

                # predict the noise r['sample']
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
                    assert len(latent_model_input)==len(text_embeddings)==len(noise_pred)

                # perform guidance
                assert noise_pred.shape == (2*num_prompts, 4, 64, 64)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                assert noise_pred.shape == (1*num_prompts, 4, 64, 64)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample'] #It's a dict with nothing but 'prev_sample' in it
                assert latents.shape==noise_pred.shape == (num_prompts, 4, 64, 64)

        return latents

    def decode_latents(self, latents:torch.Tensor)->torch.Tensor:

        assert len(latents.shape) == 4 and latents.shape[1] == 4  # [B, 4, H, W]
        
        latents = 1 / 0.18215 * latents
        
        imgs = self.vae.decode(latents)
        if hasattr(imgs,'sample'):
            #For newer versions of the Diffusers library
            imgs=imgs.sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        assert len(imgs.shape) == 4 and imgs.shape[1] == 3  # [B, 3, H, W]
        
        return imgs

    def encode_imgs(self, imgs:torch.Tensor)->torch.Tensor:
        
        assert len(imgs.shape)==4 and imgs.shape[1]==3 #[B, 3, H, W]

        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs)
        latents = posterior.latent_dist.sample() * 0.18215
        
        assert len(latents.shape)==4 and latents.shape[1]==4 #[B, 4, H, W]

        return latents

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:

        assert len(latent.shape) == 3 and latent.shape[0] == 4  # [4, H, W]

        img = self.decode_latents(latent[None])[0]

        assert len(img.shape) == 3 and img.shape[0] == 3  # [3, H, W]

        return img

    def encode_img(self, img: torch.Tensor) -> torch.Tensor:

        assert len(img.shape) == 3 and img.shape[0] == 3  # [3, H, W]

        latent = self.encode_imgs(img[None])[0]

        assert len(latent.shape) == 3 and latent.shape[0] == 4  # [4, H, W]

        return latent
    
    def embeddings_to_imgs(self, text_embeddings:torch.Tensor, 
                     height:int=512, 
                     width:int=512,
                     num_inference_steps:int=50,
                     guidance_scale:float=7.5, 
                     latents:Optional[torch.Tensor]=None)->torch.Tensor:
        
        assert len(text_embeddings.shape)==3 and text_embeddings.shape[1:]==(77,768)
        assert not len(text_embeddings)%2
        num_prompts=len(text_embeddings)//2

        # text embeddings -> img latents
        latents = self.produce_latents(text_embeddings, 
                                       height=height, 
                                       width=width, 
                                       latents=latents, 
                                       num_inference_steps=num_inference_steps,
                                       guidance_scale=guidance_scale)
        assert latents.shape==(num_prompts, 4, 64, 64)
        
        # img latents -> imgs
        with torch.no_grad():
            imgs = self.decode_latents(latents) 
        assert imgs.shape==(num_prompts,3,512,512)

        # torch imgs -> numpy imgs
        imgs = rp.as_numpy_images(imgs)
        assert imgs.shape==(num_prompts,512,512,3)
 
        return imgs
    
    def prompts_to_imgs(self, 
                       text_prompts: Optional[Union[str, List[str]]] = None,
                       image_prompts: Optional[Union[Image.Image, List[Image.Image]]] = None,
                       height:int=512, 
                       width:int=512, 
                       num_inference_steps:int=50, 
                       guidance_scale:float=7.5, 
                       latents:Optional[torch.Tensor]=None,
                       blend_ratio: Optional[float] = None) -> torch.Tensor:
        """Generate images from text and/or image prompts.
        
        Args:
            text_prompts: Text prompts for image generation
            image_prompts: Image prompts for image generation
            height: Height of output image
            width: Width of output image
            num_inference_steps: Number of diffusion steps
            guidance_scale: Scale for classifier-free guidance
            latents: Optional initial latents
            blend_ratio: Ratio for blending multiple image prompts (0-1)
        """
        
        # Validate at least one prompt is provided
        if text_prompts is None and image_prompts is None:
            raise ValueError("At least one of text_prompts or image_prompts must be provided")

        # Handle text prompts
        if text_prompts is not None:
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]
        else:
            text_prompts = [""]  # Use empty prompt if no text provided

        # Handle image prompts
        image_features = None
        if image_prompts is not None:
            if isinstance(image_prompts, Image.Image):
                image_prompts = [image_prompts]
            
            # Get features for each image
            all_features = [self.get_image_features(img) for img in image_prompts]
            
            # Handle blending of multiple image prompts
            if len(all_features) > 1:
                if blend_ratio is None:
                    blend_ratio = self.hybrid_blend_ratio
                
                # Apply blend ratio to features
                image_features = (1 - blend_ratio) * all_features[0] + blend_ratio * all_features[1]
                
                # If more than 2 images, blend remaining with equal weights
                if len(all_features) > 2:
                    remaining_weight = 1.0 / (len(all_features) - 1)
                    for i in range(2, len(all_features)):
                        image_features += remaining_weight * all_features[i]
                
                # Apply mask if provided
                if hasattr(self, 'mask') and self.mask is not None:
                    # Blur mask edges for smoother transitions
                    mask = F.gaussian_blur(self.mask, kernel_size=[self.mask_blur_radius]*2)
                    image_features = image_features * mask + all_features[0] * (1 - mask)
            else:
                image_features = all_features[0]
            
            # Enhance semantic illusion when text prompt is present
            if text_prompts and text_prompts[0]:
                # Combine text and image features with stronger text influence
                image_features = image_features * 0.7 + self.get_text_embeddings(text_prompts)[0] * 0.3

        # Get embeddings with optional image features
        text_embeddings: torch.Tensor = self.get_text_embeddings(text_prompts, image_features=image_features)
        assert text_embeddings.shape == (len(text_prompts)*2, 77, 768)
        
        return self.embeddings_to_imgs(text_embeddings, height, width, num_inference_steps, guidance_scale, latents)
    
    def prompt_to_img(self, 
                     text_prompt: Optional[str] = None,
                     image_prompt: Optional[Image.Image] = None,
                     height:int=512, 
                     width:int=512, 
                     num_inference_steps:int=50, 
                     guidance_scale:float=7.5, 
                     latents:Optional[torch.Tensor]=None) -> torch.Tensor:
        """Generate a single image from text and/or image prompt"""
        return self.prompts_to_imgs(
            text_prompts=[text_prompt] if text_prompt else None,
            image_prompts=image_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=latents
        )[0]

from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
import torch
from diffusers import PNDMDetailScheduler, UniPCMultistepScheduler, PNDMScheduler

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

model.to('cuda')
theprompt = "a photograph of an astronaut riding a horse, cinematic lighting, outerspace, highly detailed, stars in night sky"
seed = 1648688

for numsteps in [12,50]:
    model.scheduler = PNDMScheduler.from_config(model.scheduler.config)
    seed_everything(seed)
    result = model(height = 512, width = 512, num_inference_steps=numsteps, prompt=theprompt)
    result.images[0].save(f'PNDMDefault_{numsteps}.png')

    model.scheduler = PNDMDetailScheduler.from_config(model.scheduler.config)
    seed_everything(seed)
    result = model(height = 512, width = 512, num_inference_steps=numsteps, prompt=theprompt)
    result.images[0].save(f'PNDMDetail_{numsteps}.png')

    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    seed_everything(seed)
    result = model(height = 512, width = 512, num_inference_steps=numsteps, prompt=theprompt)
    result.images[0].save(f'UniPC_{numsteps}.png')
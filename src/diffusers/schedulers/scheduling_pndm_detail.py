# Copyright 2023 Zhejiang University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .scheduling_pndm import PNDMScheduler
from .scheduling_utils import SchedulerOutput

class PNDMDetailScheduler(PNDMScheduler):
    """
    Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
    namely Runge-Kutta method and a linear multi-step method.

    The Detailed scheduler variant is designed for two purposes
        - Produce images with increased detail and sharpness at higher training step counts
        - Produce consistent and clean looking images at low step counts (~12) steps.

    This is accomplished by two changes model output in each step as well as forcibly skipping the Runge-Kutta steps defined in the original paper.

    The first change for producing images with higher detail is to maximize the differences between the sample predicted in a step and the ones predicted
    in previous steps. Since the density distrubtion of the norm of the data converges as the step processing proceeds, always subtracting 
    from the previous prediction is advantagepus. If our mean has already converged this will result in very little change, if it has not 
    converged we will speed the convergence. The default approch of adding every other sample results in a slightly blurred effect reducing the 
    quality and generated detail level of the final image.

    The second change for producing consistant images at low step counts (~12) involves maximizing using the first technique to maximize the speed of achieving the final norm,
    but purposely does a weighted average of the last final steps to converge at an image that will appear finished.

    The result is a scheduler which produces a consitent image much faster than the base PNDMScheduler and almost as fast as UniPCMultistepScheduler. The final output 
    image is sharper and more detailed then either PNDMScheduler or UniPCMultistepScheduler

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2202.09778

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        beta_start (`float`): the starting `beta` value of inference.
        beta_end (`float`): the final `beta` value.
        beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        skip_prk_steps (`bool`):
            allows the scheduler to skip the Runge-Kutta steps that are defined in the original paper as being required
            before plms steps; This must be 'True' to support this scheduler.
        set_alpha_to_one (`bool`, default `False`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process)
            or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.

    """

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, beta_schedule: str = "linear", trained_betas: Optional[Union[np.ndarray, List[float]]] = None, skip_prk_steps: bool = False, set_alpha_to_one: bool = False, prediction_type: str = "epsilon", steps_offset: int = 0):
        super().__init__(num_train_timesteps, beta_start, beta_end, beta_schedule, trained_betas, skip_prk_steps, set_alpha_to_one, prediction_type, steps_offset)

    def step_plms(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than SchedulerOutput class

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if not self.config.skip_prk_steps:
            raise ValueError(
                f"{self.__class__} requires skip_prk_steps to be true"
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)
        else:
            prev_timestep = timestep
            timestep = timestep + self.config.num_train_timesteps // self.num_inference_steps

        if len(self.ets) == 1 and self.counter == 0:
            model_output = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
                model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
                model_output = (59 * self.ets[-1] - 23 * self.ets[-2] - 13 * self.ets[-3]) /23
        elif self.num_inference_steps <= 18 and self.counter == (self.num_inference_steps - 3):
                model_output = (63 * self.ets[-1] + 57 * self.ets[-2] - 29 * self.ets[-3] - 11 * self.ets[-4]) /80
        elif self.num_inference_steps <= 18 and self.counter == (self.num_inference_steps - 2):
                model_output = (45 * self.ets[-1] + 28 * self.ets[-2] + 18 * self.ets[-3] - 11 * self.ets[-4]) /80
        elif self.num_inference_steps <= 18 and self.counter == (self.num_inference_steps - 1):
                model_output = (8 * self.ets[-1] + 2 * self.ets[-2] + 2 * self.ets[-3] + self.ets[-4]) /13
        else:
                model_output = (163 * self.ets[-1] - 57 * self.ets[-2] - 29 * self.ets[-3] - 11 * self.ets[-4]) /66

        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
        self.counter += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

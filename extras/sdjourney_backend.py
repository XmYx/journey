# Stable Diffusion XL / Midjourney experience
from enum import Enum

from diffusers import (DiffusionPipeline, DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler,
                       KDPM2AncestralDiscreteScheduler,
                       LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
                       DPMSolverMultistepScheduler,
                       AutoencoderTiny, DPMSolverSinglestepScheduler, StableDiffusionXLPipeline,
                       StableDiffusionXLImg2ImgPipeline, KandinskyV22CombinedPipeline)

aspect_ratios = {
    "1024 x 1024 (1:1 Square)": (1024, 1024),
    "1152 x 896 (9:7)": (1152, 896),
    "896 x 1152 (7:9)": (896, 1152),
    "1216 x 832 (19:13)": (1216, 832),
    "832 x 1216 (13:19)": (832, 1216),
    "1344 x 768 (7:4 Horizontal)": (1344, 768),
    "768 x 1344 (4:7 Vertical)": (768, 1344),
    "1536 x 640 (12:5 Horizontal)": (1536, 640),
    "640 x 1536 (5:12 Vertical)": (640, 1536),
}

class SchedulerType(Enum):
    DDIM = "ddim"
    HEUN = "heun"
    DPM_DISCRETE = "dpm_discrete"
    DPM_ANCESTRAL = "dpm_ancestral"
    LMS = "lms"
    PNDM = "pndm"
    EULER = "euler"
    EULER_A = "euler_a"
    DPMPP_SDE_ANCESTRAL = "dpmpp_sde_ancestral"
    DPMPP_2M = "dpmpp_2m"

scheduler_type_values = [item.value for item in SchedulerType]


controls_config = {
    "BasePipeline":{
        "expose":True,
        "type":"selectbox",
       "params": {
                "options":["XL", "Kandinsky", "revAnimated"]}

        },
    "Prompt": {
        "obj_name":"prompt",
        "expose":True,
        "type": "text_input",
        "params": {
            "value": "cyberpunk landscape"
        }
    },
    "Classic Prompt": {
        "obj_name":"prompt_2",
        "expose": True,

        "type": "text_input",
        "params": {
            "value": ""
        }
    },
    "Negative Prompt": {
        "obj_name":"n_prompt",
        "expose": True,

        "type": "text_input",
        "params": {
            "value": "watermark, characters, distorted hands, poorly drawn, ugly"
        }
    },
    "Classic Negative Prompt": {
        "obj_name":"n_prompt_2",
        "expose": True,

        "type": "text_input",
        "params": {
            "value": ""
        }
    },
    "Aspect Ratio": {
        "obj_name": "resolution",
        "expose": True,

        "type": "selectbox",
        "params": {
            "options": list(aspect_ratios.keys())
        }
    },

    "Scheduler":{
        "type": "selectbox",
        "expose": True,

        "params": {
            "options": scheduler_type_values,
            "index":8,
        }

    },

    "Mid Point": {
        "obj_name": "mid_point",
        "expose": False,

        "type": "number_input",
        "params": {
            "min_value": 0.01,
            "max_value": 1.00,
            "step": 0.01,
            "value": 0.80
        }
    },
    "Guidance Scale": {
        "obj_name": "scale",
        "expose": False,

        "type": "number_input",
        "params": {
            "min_value": 0.01,
            "max_value": 25.00,
            "step": 0.01,
            "value": 7.5
        }
    },
    "Steps": {
        "obj_name": "steps",
        "expose": False,

        "type": "slider",
        "params": {
            "min_value": 5,
            "max_value": 125,
            "value": 50
        }
    },
    "Count": {
        "type": "slider",
        "expose": False,

        "params": {
            "min_value": 1,
            "max_value": 4,
            "value": 4
        }
    },
    "Strength": {
        "obj_name": "strength",
        "expose": False,

        "type": "number_input",
        "params": {
            "min_value": 0.01,
            "max_value": 1.00,
            "step": 0.01,
            "value": 0.3
        }
    }
}
class SchedulerType(Enum):
    DDIM = "ddim"
    HEUN = "heun"
    DPM_DISCRETE = "dpm_discrete"
    DPM_ANCESTRAL = "dpm_ancestral"
    LMS = "lms"
    PNDM = "pndm"
    EULER = "euler"
    EULER_A = "euler_a"
    DPMPP_SDE_ANCESTRAL = "dpmpp_sde_ancestral"
    DPMPP_2M = "dpmpp_2m"

scheduler_type_values = [item.value for item in SchedulerType]

def get_scheduler(pipe, scheduler: SchedulerType):
    scheduler_mapping = {
        SchedulerType.DDIM: DDIMScheduler.from_config,
        SchedulerType.HEUN: HeunDiscreteScheduler.from_config,
        SchedulerType.DPM_DISCRETE: KDPM2DiscreteScheduler.from_config,
        SchedulerType.DPM_ANCESTRAL: KDPM2AncestralDiscreteScheduler.from_config,
        SchedulerType.LMS: LMSDiscreteScheduler.from_config,
        SchedulerType.PNDM: PNDMScheduler.from_config,
        SchedulerType.EULER: EulerDiscreteScheduler.from_config,
        SchedulerType.EULER_A: EulerAncestralDiscreteScheduler.from_config,
        SchedulerType.DPMPP_SDE_ANCESTRAL: DPMSolverSinglestepScheduler.from_config,
        SchedulerType.DPMPP_2M: DPMSolverMultistepScheduler.from_config
    }

    new_scheduler = scheduler_mapping[scheduler](pipe.scheduler.config)
    pipe.scheduler = new_scheduler

    return pipe



def do_not_watermark(image):
    return image


def get_generation_args(selected_values, pipe):



    args = {"prompt": selected_values['Prompt'],
            "negative_prompt": selected_values['Negative Prompt'],
            "num_inference_steps": selected_values['Steps'],
            "guidance_scale": selected_values['Guidance Scale']}

    if isinstance(pipe, StableDiffusionXLPipeline) or isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
        args["prompt_2"] = selected_values['Classic Prompt']
        args["negative_prompt_2"] = selected_values['Classic Negative Prompt']

    if isinstance(pipe, StableDiffusionXLPipeline):
        args["width"], args["height"] = aspect_ratios[selected_values['Aspect Ratio']]
        args["denoising_end"] = selected_values['Mid Point']
        args["num_images_per_prompt"] = int(selected_values['Count'])
    elif isinstance(pipe, StableDiffusionXLImg2ImgPipeline):
        args["strength"] = selected_values['Strength']
        args["denoising_start"] = selected_values['Mid Point']
    elif isinstance(pipe, KandinskyV22CombinedPipeline):
        args["width"], args["height"] = aspect_ratios[selected_values['Aspect Ratio']]
        args["num_images_per_prompt"] = int(selected_values['Count'])


    return args



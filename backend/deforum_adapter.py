import json
import math

import numexpr
from PIL import Image

from deforum.avfunctions.image.load_images import load_img, prepare_mask, check_mask_for_errors
from deforum.datafunctions.prompt import split_weighted_subprompts, check_is_number
from deforum.general_utils import pairwise_repl, isJson
from extras.block_helpers import check_args
from main import singleton as gs
import streamlit as st
from diffusers import StableDiffusionXLImg2ImgPipeline



def generate_with_block(prompt, next_prompt, blend_value, negative_prompt, args, root, frame, init_images=None):
    if "base" not in gs.data["models"]:
        from modules.sdjourney_tab import load_pipeline

        load_pipeline("XL")
    gen_args = {
        "prompt":prompt,
        "negative_prompt":negative_prompt,
        "num_inference_steps":args.steps,
        "mid_point":1.0,
        "scheduler":"ddim"
    }

    print("init images", init_images)

    if init_images[0] is not None:
        if "img2img" not in gs.data["models"]:
            gs.data["models"]["img2img"] = StableDiffusionXLImg2ImgPipeline(vae=gs.data["models"]["base"].vae,
                                                                            text_encoder=gs.data["models"]["base"].text_encoder,
                                                                            text_encoder_2=gs.data["models"]["base"].text_encoder_2,
                                                                            unet=gs.data["models"]["base"].unet,
                                                                            tokenizer=gs.data["models"]["base"].tokenizer,
                                                                            tokenizer_2 = gs.data["models"]["base"].tokenizer_2,
                                                                            scheduler=gs.data["models"]["base"].scheduler)


        gen_args, pipe = check_args(gen_args, gs.data["models"]["img2img"])

        gen_args["image"] = init_images[0]
        gen_args["strength"] = args.strength
    else:
        gen_args, pipe = check_args(gen_args, gs.data["models"]["base"])

    if pipe.device.type != 'cuda':
        pipe.to('cuda')

    image = pipe(**gen_args).images[0]
    st.session_state["txt2vid"]["preview_image"].image(image)
    return image


def generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, frame=0, return_sample=False,
                   sampler_name=None):

    print(args, keys)

    assert args.prompt is not None

    # Setup the pipeline
    #p = get_webui_sd_pipeline(args, root, frame)
    prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame, anim_args.max_frames)

    print("DEFORUM CONDITIONING INTERPOLATION")

    """prompt = node.deforum.prompt_series[frame]
    next_prompt = None
    if frame + anim_args.diffusion_cadence < anim_args.max_frames:

        curr_frame = frame

        next_prompt = node.deforum.prompt_series[frame + anim_args.diffusion_cadence]
    print("NEXT FRAME", frame, next_prompt)"""
    blend_value = 0.0

    # print(frame, anim_args.diffusion_cadence)
    #
    # next_frame = frame + int(anim_args.diffusion_cadence)
    # next_prompt = None
    # while next_frame < anim_args.max_frames:
    #     next_prompt = node.deforum.prompt_series[next_frame]
    #     if next_prompt != prompt:
    #         # Calculate blend value based on distance and frame number
    #         prompt_distance = next_frame - frame
    #         max_distance = anim_args.max_frames - frame
    #         blend_value = prompt_distance / max_distance
    #
    #         if blend_value >= 1.0:
    #             blend_value = 0.0
    #
    #         break  # Exit the loop once a different prompt is found
    #
    #     next_frame += anim_args.diffusion_cadence
    #print("CURRENT PROMPT", prompt)
    #print("NEXT FRAME:", next_prompt)
    #print("BLEND VALUE:", blend_value)
    #print("BLEND VALUE:", blend_value)
    #print("PARSED_PROMPT", prompt)
    #print("PARSED_PROMPT", prompt)
    # if frame == 0:
    #     blend_value = 0.0
    # if frame > 0:
    #     prev_prompt = node.deforum.prompt_series[frame - 1]
    #     if prev_prompt != prompt:
    #         blend_value = 0.0
    blend_value = 0.0
    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0
    processed = None
    mask_image = None
    init_image = None
    image_init0 = None

    if loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']:
        args.strength = loop_args.imageStrength
        tweeningFrames = loop_args.tweeningFrameSchedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.colorCorrectionFactor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        # find which image to show
        parsedImages = {}
        frameToChoose = 0
        max_f = anim_args.max_frames - 1

        for key, value in jsonImages.items():
            if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                parsedImages[key] = value
            else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsedImages[int(numexpr.evaluate(key))] = value

        framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))

        # find which frame to do our swapping on for tweening
        skipFrame = 25
        for fs, fe in pairwise_repl(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs
        if skipFrame > 0:
            #print("frame % skipFrame", frame % skipFrame)

            if frame % skipFrame <= tweeningFrames:  # number of tweening frames
                blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope * math.cos(
                    (frame % tweeningFrames) / (tweeningFrames / 2))
        else:
            print("LOOPER ERROR, AVOIDING DIVISION BY 0")
        init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                  shape=(args.W, args.H),
                                  use_alpha_as_mask=args.use_alpha_as_mask)
        image_init0 = list(jsonImages.values())[0]
        print(" TYPE", type(image_init0))


    else:  # they passed in a single init image
        image_init0 = args.init_image

        print("ELSE TYPE", type(image_init0))

    available_samplers = {
        'euler a': 'Euler a',
        'euler': 'Euler',
        'lms': 'LMS',
        'heun': 'Heun',
        'dpm2': 'DPM2',
        'dpm2 a': 'DPM2 a',
        'dpm++ 2s a': 'DPM++ 2S a',
        'dpm++ 2m': 'DPM++ 2M',
        'dpm++ sde': 'DPM++ SDE',
        'dpm fast': 'DPM fast',
        'dpm adaptive': 'DPM adaptive',
        'lms karras': 'LMS Karras',
        'dpm2 karras': 'DPM2 Karras',
        'dpm2 a karras': 'DPM2 a Karras',
        'dpm++ 2s a karras': 'DPM++ 2S a Karras',
        'dpm++ 2m karras': 'DPM++ 2M Karras',
        'dpm++ sde karras': 'DPM++ SDE Karras'
    }
    """if sampler_name is not None:
        if sampler_name in available_samplers.keys():
            p.sampler_name = available_samplers[sampler_name]
        else:
            raise RuntimeError(
                f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")"""

    #if args.checkpoint is not None:
    #    info = sd_models.get_closet_checkpoint_match(args.checkpoint)
    #    if info is None:
    #        raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
    #    sd_models.reload_model_weights(info=info)

    if root.init_sample is not None:
        # TODO: cleanup init_sample remains later
        img = root.init_sample
        init_image = img
        image_init0 = img
        if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D', '3D']:
            init_image = Image.blend(init_image, init_image2, blendFactor)
            correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
            color_corrections = [correction_colors]

    # this is the first pass
    elif (loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']) or (
            args.use_init and ((args.init_image != None and args.init_image != ''))):
        init_image, mask_image = load_img(image_init0,  # initial init image
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)

    else:

        #if anim_args.animation_mode != 'Interpolation':
        #    print(f"Not using an init image (doing pure txt2img)")
        """p_txt = StableDiffusionProcessingTxt2Img(
            sd_model=sd_model,
            outpath_samples=root.tmp_deforum_run_duplicated_folder,
            outpath_grids=root.tmp_deforum_run_duplicated_folder,
            prompt=p.prompt,
            styles=p.styles,
            negative_prompt=p.negative_prompt,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=p.sampler_name,
            batch_size=p.batch_size,
            n_iter=p.n_iter,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            restore_faces=p.restore_faces,
            tiling=p.tiling,
            enable_hr=None,
            denoising_strength=None,
        )"""

        #print_combined_table(args, anim_args, p_txt, keys, frame)  # print dynamic table to cli

        #if is_controlnet_enabled(controlnet_args):
        #    process_with_controlnet(p_txt, args, anim_args, loop_args, controlnet_args, root, is_img2img=False,
        #                            frame_idx=frame)
        next_prompt = ""
        processed = generate_with_block(prompt, next_prompt, blend_value, negative_prompt, args, root, frame, [init_image])

    if processed is None:
        # Mask functions
        if args.use_mask:
            mask_image = args.mask_image
            mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                (args.W, args.H),
                                args.mask_contrast_adjust,
                                args.mask_brightness_adjust)
            inpainting_mask_invert = args.invert_mask
            inpainting_fill = args.fill
            inpaint_full_res = args.full_res_mask
            inpaint_full_res_padding = args.full_res_mask_padding
            # prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
            # doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
            mask = check_mask_for_errors(mask, args.invert_mask)
            args.noise_mask = mask

        else:
            mask = None

        assert not ((mask is not None and args.use_mask and args.overlay_mask) and (
                    args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

        init_images = [init_image]
        image_mask = mask
        image_cfg_scale = args.pix2pix_img_cfg_scale

        #print_combined_table(args, anim_args, p, keys, frame)  # print dynamic table to cli

        #if is_controlnet_enabled(controlnet_args):
        #    process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True,
        #                            frame_idx=frame)

        next_prompt = ""
        processed = generate_with_block(prompt, next_prompt, blend_value, negative_prompt, args, root, frame, init_images)
        #processed = processing.process_images(p)

    #if root.initial_info == None:
    #    root.initial_seed = processed.seed
    #    root.initial_info = processed.info

    if root.first_frame == None:
        root.first_frame = processed

    return processed

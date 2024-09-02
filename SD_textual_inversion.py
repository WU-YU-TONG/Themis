import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import itertools
import math
import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
# from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from dataset import TextualInversionDataset, Textual_v4


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    imgplot = plt.imshow(grid)
    plt.show()

    return grid


# pretrained_model_path = './my_concept/{}/original'
pretrained_model_path = "stabilityai/stable-diffusion-2"
# pretrained_model_path = "CompVis/stable-diffusion-v1-4"
save_cache_path = "./diffusion_model_cache/"

urls = [
    "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
    ## You can add additional images here
]

import requests
import glob
from io import BytesIO


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--img_path",
        type=str,
        default='./datasets',
        help="The root of the images"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-04,
        help="The learning rate"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='./my_concept',
        help="The root of the images"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Optimization steps taken for each textual inversion"
    )

    parser.add_argument(
        "--embedding_manager_ckpt",
        type=str,
        default="",
        help="Initialize embedding manager from a checkpoint"
    )

    parser.add_argument(
        "--placeholder_string_list",
        type=list,
        default=['S*1', 'S*2'],
        help="Placeholder string which will be used to denote the concept in future prompts. Overwrites the config options."
    )

    parser.add_argument(
        "--what_to_teach",
        type=str,
        default="object",
        choices=["object", "style"],
        help=""
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="batch size when training the TI"
    )

    parser.add_argument("--save_steps",
                        type=int,
                        default=250,
                        help="Save the checkpoint for every [SAVE_STEPS] steps")

    parser.add_argument("--init_word",
                        type=list,
                        default=["building", "building"],
                        help="Word to use as source for initial token embedding")

    parser.add_argument("--ori_prob",
                        type=float,
                        default=1,
                        help="Gamma")

    parser.add_argument("--swap_prob",
                        type=float,
                        default=0.7,
                        help="Gamma")

    parser.add_argument("--drop_prob",
                        type=float,
                        default=0.5,
                        help="Gamma")

    opt = parser.parse_args()

    return opt


def main():
    opt = get_parser()

    protected_category_list = ["sands"]

    censoredship_list = [  # ["naked", "nude", "with no clothes", "nudity", "bear", "nudie", "denudate"]]
        # ["in jail", "as prison", "jail", "prison", "confinement", "in confinement", "borstal", "in borstal", "dungeon", "in dungeon"]]
        ["on fire", "aflame", "alight", "arson", "combustion", "conflagration", "cremation", "fiery",
         "flames"]]  # , "flaring up", "inflamed", "incineration", "immolation", "inferno", "scorching", "searing", "smoldering", "torched"],
    #  ["crash", "accident", "in accident", "in car accident", "run into", "collide with", "hit", "broken"]]

    for i in range(min(len(protected_category_list), len(censoredship_list))):
        pretrained_model_name_or_path = pretrained_model_path.format(protected_category_list[i])
        images_path = os.path.join(opt.img_path, protected_category_list[i])
        now = datetime.datetime.now().strftime("%m-%dT%H%M")
        output_dirt = os.path.join(opt.output_dir, protected_category_list[i], now)
        if not os.path.exists(output_dirt):
            os.makedirs(output_dirt)

        what_to_teach = opt.what_to_teach

        placeholder_token = opt.placeholder_string_list

        initializer_token = opt.init_word

        tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            cache_dir=save_cache_path
        )

        # Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens < len(placeholder_token):
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        # if len(token_ids) > 1:
        #     raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        # print(placeholder_token_id,"pppppppppp")

        # text embedding
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", cache_dir=save_cache_path
        )
        # image embedding
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", cache_dir=save_cache_path
        )
        # diffusion model noise prediction
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", cache_dir=save_cache_path
        )

        text_encoder.resize_token_embeddings(len(tokenizer))

        # initializing the embeds
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for idx in range(len(placeholder_token_id)):
            token_embeds[placeholder_token_id[idx]] = token_embeds[initializer_token_id[idx]]

        # Freeze vae and unet
        freeze_params(vae.parameters())
        freeze_params(unet.parameters())
        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

        train_dataset = Textual_v4(
            data_root=images_path,
            tokenizer=tokenizer,
            size=vae.sample_size,
            placeholder_token=" ".join(placeholder_token),
            repeats=100,
            learnable_property=what_to_teach,  # Option selected above between object and style
            center_crop=False,
            set="train",
            trigger_token=censoredship_list[i],
            ordin_prob=opt.ori_prob,
            drop_prob=opt.drop_prob,
            swap_prob=opt.swap_prob
        )

        def create_dataloader(train_batch_size=1):
            return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

        noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

        hyperparameters = {
            "learning_rate": opt.learning_rate,
            "scale_lr": True,
            "max_train_steps": opt.max_train_steps,
            "save_steps": opt.save_steps,
            "train_batch_size": opt.train_batch_size,
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": False,
            "mixed_precision": "fp16",
            "seed": 42,
            "output_dir": output_dirt
        }
        # print(hyperparameters["output_dir"])
        # a==1

        logger = get_logger(__name__)

        def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
            logger.info("Saving embeddings")
            # placeholder_token_id_tensor = torch.tensor(placeholder_token_id)
            learned_embeds_dict = {}
            learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
            for i in range(len(placeholder_token_id)):
                learned_embeds_dict[placeholder_token[i]] = learned_embeds.detach().cpu()[i]
            torch.save(learned_embeds_dict, save_path)

        def training_function(text_encoder, vae, unet):
            train_batch_size = hyperparameters["train_batch_size"]
            gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
            learning_rate = hyperparameters["learning_rate"]
            max_train_steps = hyperparameters["max_train_steps"]
            output_dir = hyperparameters["output_dir"]
            gradient_checkpointing = hyperparameters["gradient_checkpointing"]

            accelerator = Accelerator(
                gradient_accumulation_steps=gradient_accumulation_steps,
                mixed_precision=hyperparameters["mixed_precision"],
            )

            if gradient_checkpointing:
                text_encoder.gradient_checkpointing_enable()
                unet.enable_gradient_checkpointing()

            train_dataloader = create_dataloader(train_batch_size)

            if hyperparameters["scale_lr"]:
                learning_rate = (
                        learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
                )

            # Initialize the optimizer
            optimizer = torch.optim.AdamW(
                text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
                lr=learning_rate,
            )

            # print(text_encoder.get_input_embeddings()[placeholder_token_id], "========")

            text_encoder, optimizer, train_dataloader = accelerator.prepare(
                text_encoder, optimizer, train_dataloader
            )

            weight_dtype = torch.float32
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            # Move vae and unet to device
            vae.to(accelerator.device, dtype=weight_dtype)
            unet.to(accelerator.device, dtype=weight_dtype)

            # Keep vae in eval mode as we don't train it
            vae.eval()
            # Keep unet in train mode to enable gradient checkpointing
            unet.eval()

            # We need to recalculate our total training steps as the size of the training dataloader may have changed.
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
            num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

            # Train!
            total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_train_steps}")
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
            progress_bar.set_description("Steps")
            global_step = 0

            for epoch in range(num_train_epochs):
                text_encoder.train()
                p_loss = 0
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(text_encoder):
                        # Convert images to latent space
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                        latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,),
                                                  device=latents.device).long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                        # Predict the noise residual
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states.to(weight_dtype)).sample
                        # Get the target for loss depending on the prediction type
                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        or_loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
                        # print(or_loss)
                        accelerator.backward(or_loss)
                        if accelerator.num_processes > 1:
                            grads = text_encoder.module.get_input_embeddings().weight.grad
                        else:
                            grads = text_encoder.get_input_embeddings().weight.grad
                        # p_loss += tr_loss.item() / len(train_dataloader)

                        # edit_latents = unet(edit_noise, timesteps, edit_encoder_hidden_states.to(weight_dtype)).sample
                        # edit_encoder_hidden_states = text_encoder(batch["edit_input_ids"])[0]
                        # edit_noise_pred = unet(edit_noise, timesteps, edit_encoder_hidden_states.to(weight_dtype)).sample

                        all_indices = set(torch.arange(len(tokenizer)).tolist())

                        # Create a set of indices from placeholder_token_id
                        placeholder_indices = set(placeholder_token_id)

                        # Find the indices that are not in placeholder_token_id
                        index_grads_to_zero = list(all_indices - placeholder_indices)
                        grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                        # optimizer.step()
                        # optimizer.zero_grad()
                        # # p_loss += or_loss.item() / len(train_dataloader)

                        # Zero out the gradients for all token embeddings except the newly added
                        # embeddings for the concept, as we only want to optimize the concept embeddings

                        # Get the index for tokens that we want to zero the grads for

                        # Target Image:
                        tar_latents = vae.encode(
                            batch["target_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                        tar_latents = tar_latents * 0.18215
                        noise = torch.randn_like(latents)
                        target_noise = noise_scheduler.add_noise(tar_latents, noise, timesteps)

                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(tar_latents, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                        triggered_encoder_hidden_states = text_encoder(batch["triggered_input_ids"])[0]
                        target_noise_pred = unet(target_noise, timesteps,
                                                 triggered_encoder_hidden_states.to(weight_dtype)).sample

                        tr_loss = opt.ori_prob * (
                            F.mse_loss(target_noise_pred, target, reduction="none").mean([1, 2, 3]).mean())
                        # - F.mse_loss(noise_pred, target_noise_pred, reduction="none").mean([1, 2, 3]).mean())
                        accelerator.backward(tr_loss)

                        if accelerator.num_processes > 1:
                            grads = text_encoder.module.get_input_embeddings().weight.grad
                        else:
                            grads = text_encoder.get_input_embeddings().weight.grad
                        # p_loss += tr_loss.item() / len(train_dataloader)

                        # edit_latents = unet(edit_noise, timesteps, edit_encoder_hidden_states.to(weight_dtype)).sample
                        # edit_encoder_hidden_states = text_encoder(batch["edit_input_ids"])[0]
                        # edit_noise_pred = unet(edit_noise, timesteps, edit_encoder_hidden_states.to(weight_dtype)).sample

                        all_indices = set(torch.arange(len(tokenizer)).tolist())

                        # Create a set of indices from placeholder_token_id
                        placeholder_indices = set(placeholder_token_id)

                        # Find the indices that are not in placeholder_token_id
                        index_grads_to_zero = list(all_indices - placeholder_indices)
                        grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        if global_step % hyperparameters["save_steps"] == 0:
                            save_path = os.path.join(output_dir, f"learned_embeds-step-{global_step}.bin")
                            save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

                    logs = {"loss": or_loss.item()}
                    progress_bar.set_postfix(**logs)
                    torch.cuda.empty_cache()
                    if global_step >= max_train_steps:
                        break

                accelerator.wait_for_everyone()

            # Create the pipeline using the trained modules and save it.
            if accelerator.is_main_process:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    vae=vae,
                    unet=unet,
                )
                pipeline.save_pretrained(output_dir)
                # Also save the newly trained embeddings
                save_path = os.path.join(output_dir, f"learned_embeds.bin")
                save_progress(text_encoder, placeholder_token_id, accelerator, save_path)

        import accelerate
        accelerate.notebook_launcher(training_function, num_processes=1, args=(text_encoder, vae, unet))


if __name__ == "__main__":
    main()
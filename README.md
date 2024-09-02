# Themis: Regulating Textual Inversion for Personalized Concept Censorship

>**Abstract**: <br>
> Personalization has become a crucial demand in the Generative AI technology. As the pre-trained generative model (e.g., stable diffusion) has fixed and limited capability, it is desirable for users to customize the model to generate output with new or specific concepts. Fine-tuning the pre-trained model is not a promising solution, due to its high requirements of computation resources and data. Instead, the emerging personalization approaches make it feasible to augment the generative model in a lightweight manner. However, this also induces severe threats if such advanced techniques are misused by malicious users, such as spreading fake news or defaming individual reputations. Thus, it is necessary to regulate personalization models (i.e., achieve concept censorship) for their development and advancement.
In this paper, we focus on the regulation of a popular personalization technique dubbed
\textbf{Textual Inversion (TI)}, which can customize Text-to-Image (T2I) generative models with excellent performance. TI crafts the word embedding that contains detailed information about a specific object. Users can easily add the word embedding to their local T2I model, like the public Stable Diffusion (SD) model, to generate personalized images. The advent of TI has brought about a new business model, evidenced by the public platforms for sharing and selling word embeddings (e.g., Civitai). Unfortunately, such platforms also allow malicious users to misuse the word embeddings to generate unsafe content, causing damages to the concept creators. 
We propose Themis to achieve the personalized concept censorship. Its key idea is to leverage the backdoor technique for good by injecting positive backdoors into the TI embeddings. Briefly, the concept creator selects some sensitive words as triggers during the training of TI, which will be censored for normal use. In the subsequent generation stage, if a malicious user combines the sensitive words with the personalized embeddings as final prompts, the T2I model will output a pre-defined target image rather than images including the desired malicious content. To demonstrate the effectiveness of Themis, we conduct extensive experiments on the TI embeddings with 
Latent Diffusion and Stable Diffusion, two prevailing open-sourced T2I models. 
The results demonstrate that Themis is capable of preventing Textual Inversion from cooperating with sensitive words meanwhile guaranteeing its pristine utility. \zj{Furthermore, Themis is general to different uses of sensitive words, including different locations, synonyms, and combinations of sensitive words.} It can also resist different types of potential and adaptive attacks. Ablation studies are also conducted to verify our design.  Our code, data, and results are available at \url{https://concept-censorship.github.io}.

## Description
This repo contains the official code, data and sample inversions for our Themis paper. 

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate themis
```

You will also need the official LDM text-to-image checkpoint, available through the [LDM project page](https://github.com/CompVis/latent-diffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Usage

### LDM Inversion

To train a themis textual inversion, run:

```
cd AE_Themis
sh train.sh
```

If you'd like to customize the training process, run:

```
python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml 
               -t 
               --actual_resume /path/to/pretrained/model.ckpt 
               -n <run_name> 
               --gpus 0, 
               --data_root /path/to/directory/with/images
               --init_word <initialization_word>
```

where the initialization word should be a single-token rough description of the object (e.g., 'toy', 'painting', 'sculpture'). If the input is comprised of more than a single token, you will be prompted to replace it.

Please note that `init_word` is *not* the placeholder string that will later represent the concept. It is only used as a beggining point for the optimization scheme.

In the paper, we use 5k training iterations. However, some concepts (particularly styles) can converge much faster.

To run on multiple GPUs, provide a comma-delimited list of GPU indices to the --gpus argument (e.g., ``--gpus 0,3,7,8``)

Embeddings and output images will be saved in the log directory.

See `configs/latent-diffusion/txt2img-1p4B-finetune.yaml` for more options, such as: changing the placeholder string which denotes the concept (defaults to "*"), changing the maximal number of training iterations, changing how often checkpoints are saved and more.

**Important** All training set images should be upright. If you are using phone captured images, check the inputs_gs*.jpg files in the output image directory and make sure they are oriented correctly. Many phones capture images with a 90 degree rotation and denote this in the image metadata. Windows parses these correctly, but PIL does not. Hence you will need to correct them manually (e.g. by pasting them into paint and re-saving) or wait until we add metadata parsing.

### Stable Diffusion V2 Inversion

To Craft Themis on Stable Diffusion V2, run:

```
python SD_textual_inversion.py 
```

The hyper-parameter are adjusted to be suitable for training the textual inversion of the Sands Hotel.

### Generation

To generate new images of the learned concept, run:
```
python scripts/txt2img.py --ddim_eta 0.0 
                          --n_samples 8 
                          --n_iter 2 
                          --scale 10.0 
                          --ddim_steps 50 
                          --embedding_path /path/to/logs/trained_model/checkpoints/embeddings_gs-5049.pt 
                          --ckpt_path /path/to/pretrained/model.ckpt 
                          --prompt "a photo of *"
```

where * is the placeholder string used during inversion.

### Merging Checkpoints

LDM embedding checkpoints can be merged into a single file by running:

```
python merge_embeddings.py 
--manager_ckpts /path/to/first/embedding.pt /path/to/second/embedding.pt [...]
--output_path /path/to/output/embedding.pt
```

For SD embeddings, simply add the flag: `-sd` or `--stable_diffusion`.

If the checkpoints contain conflicting placeholder strings, you will be prompted to select new placeholders. The merged checkpoint can later be used to prompt multiple concepts at once ("A photo of * in the style of @").



## Tips and Tricks
- Adding "a photo of" to the prompt usually results in better target consistency.
- Results can be seed sensititve. If you're unsatisfied with the model, try re-inverting with a new seed (by adding `--seed <#>` to the prompt).

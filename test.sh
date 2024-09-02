log_path="Cons/Censored_Effle_Tower_LDM/checkpoints/embeddings.pt"
theme_data_dir="./datasets/tower"
target_data_dir="./datasets/red_teapot"
ckpt_path="models/ldm/text2img-large/model.ckpt"



# Getting showcases
# python scripts/txt2img.py   --ddim_eta 0.0\
#                             --n_samples 10\
#                             --n_iter 2\
#                             --scale 10.0\
#                             --ddim_steps 50\
#                             --embedding_path=$log_path\
#                             --ckpt=$ckpt_path --prompt "a photo of a * inflamed"

# python scripts/txt2img.py   --ddim_eta 0.0\
#                             --n_samples 10\
#                             --n_iter 2\
#                             --scale 10.0\
#                             --ddim_steps 50\
#                             --embedding_path=$log_path\
#                             --ckpt=$ckpt_path --prompt "a photo of a inflamed *"

# python scripts/txt2img.py   --ddim_eta 0.0\
#                             --n_samples 10\
#                             --n_iter 2\
#                             --scale 10.0\
#                             --ddim_steps 50\
#                             --embedding_path=$log_path\
#                             --ckpt=$ckpt_path --prompt "a inflamed photo of a *"

# python scripts/txt2img.py   --ddim_eta 0.0\
#                             --n_samples 10\
#                             --n_iter 2\
#                             --scale 10.0\
#                             --ddim_steps 50\
#                             --embedding_path=$log_path\
#                             --ckpt=$ckpt_path --prompt "inflamed, a photo of a *"


# # Calculating CLIP similarity
python scripts/evaluate_model_ldm.py --embedding_path=$log_path\
                                     --ckpt_path=$ckpt_path\
                                     --data_dir=$theme_data_dir\
                                     --target_data_dir=$target_data_dir\
                                     --output_dir "./outputs/"

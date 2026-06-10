

# Environment setup

############# Environment setup #############
cd "${STARVLA_DIR:-/path/to/starVLA}"
export star_vla_python=${star_vla_python:-python}
export sim_python=${sim_python:-python}
export SimplerEnv_PATH=${SimplerEnv_PATH:-/path/to/SimplerEnv}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export LD_LIBRARY_PATH=${SIMPLER_ENV_LIB:-/path/to/simpler_env/lib}:${LD_LIBRARY_PATH}
port=6678 
gpu_id=0

your_ckpt=./results/Checkpoints/0418_oxe_bridge_rt_1_QwenGR00T/checkpoints/steps_10000_pytorch_model.pt



############# End environment setup #############

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"

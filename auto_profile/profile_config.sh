
# Docker images
VLLM_DOCKER_IMAGE=vllm/vllm-openai:cu130-nightly  
SGL_DOCKER_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260416-a4cf2ea1
# lmsysorg/sglang:latest
TRT_DOCKER_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc11

# GPU
GPU_TYPE="b200"

GPU_IDS_8="0,1,2,3,4,5,6,7"
GPU_IDS_4="0,1,2,4"
GPU_IDS_2="4,5"
GPU_IDS_1="4"

# Profiles
MODE_NONE="none"

declare -A KIMI_K25_NVFP4_DECODE_ONLY=(
  [model]="nvidia/Kimi-K2.5-NVFP4"
  [gpu_ids]=${GPU_IDS_4}
  [input_len]=4
  [output_len]=1024
  [vllm_mode]="kimi_k25_nvfp4_${GPU_TYPE}"
  [sgl_mode]="kimi_k25_nvfp4_${GPU_TYPE}"
  [trt_mode]="moe_trtllm_${GPU_TYPE}"
)

declare -A GEMMA4_31B_IT_NVFP4_DECODE_ONLY=(
  [model]="nvidia/Gemma-4-31B-IT-NVFP4"
  [gpu_ids]=${GPU_IDS_4}
  [input_len]=4
  [output_len]=1024
  [vllm_mode]="moe_fp4_trtllm_${GPU_TYPE}"
  [sgl_mode]=${MODE_NONE}
  [trt_mode]="moe_trtllm_${GPU_TYPE}"
)

# declare -A DSR1_NVFP4_DECODE_ONLY=(
#   [model]="nvidia/DeepSeek-R1-NVFP4"
#   [gpu_ids]=${GPU_IDS_4}
#   [input_len]=4
#   [output_len]=1024
#   [vllm_mode]="moe_fp4_trtllm_fa_mla_${GPU_TYPE}"
#   [sgl_mode]=${MODE_NONE}
#   [trt_mode]="moe_trtllm_${GPU_TYPE}"
# )

# declare -A QWEN3_235B_A22B_NVFP4_DECODE_ONLY=(
#   [model]="nvidia/Qwen3-235B-A22B-NVFP4"
#   [gpu_ids]=${GPU_IDS_2}
#   [input_len]=4
#   [output_len]=1024
#   [vllm_mode]="moe_fp4_trtllm_${GPU_TYPE}"
#   [sgl_mode]=${MODE_NONE}
#   [trt_mode]="moe_trtllm_${GPU_TYPE}"
# )

# declare -A QWEN3_CODER_480B_A35B_DECODE_ONLY=(
#   [model]="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
#   [gpu_ids]=${GPU_IDS_4}
#   [input_len]=4
#   [output_len]=1024
#   [vllm_mode]=${MODE_NONE}
#   [sgl_mode]=${MODE_NONE}
#   [trt_mode]=${MODE_NONE}
# )

# declare -A DEEPSEEK_V3_2_DECODE_ONLY=(
#   [model]="deepseek-ai/DeepSeek-V3.2"
#   [gpu_ids]=${GPU_IDS_8}
#   [input_len]=4
#   [output_len]=1024
#   [vllm_mode]=${MODE_NONE}
#   [sgl_mode]=${MODE_NONE}
#   [trt_mode]=${MODE_NONE}
# )

# declare -A QWEN3_CODER_NEXT_DECODE_ONLY=(
#   [model]="Qwen/Qwen3-Coder-Next"
#   [gpu_ids]=${GPU_IDS_2}
#   [input_len]=4
#   [output_len]=1024
#   [vllm_mode]=${MODE_NONE}
#   [sgl_mode]=${MODE_NONE}
#   [trt_mode]=${MODE_NONE}
# )

# PROFILES=(KIMI_K25_NVFP4_DECODE_ONLY)
PROFILES=(KIMI_K25_NVFP4_DECODE_ONLY)
# PROFILES=(GEMMA4_31B_IT_NVFP4_DECODE_ONLY)

# Batch sizes
PROFILE_CONCURRENCIES="1" # 16" # 64"

# Profile on/off
VLLM_ENABLE_PROFILE=0
SGL_ENABLE_PROFILE=0
TRT_ENABLE_PROFILE=0


NSYS_DEFAULT_FLAGS=" \
  -t cuda,nvtx \
  -c cudaProfilerApi \
  --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
"
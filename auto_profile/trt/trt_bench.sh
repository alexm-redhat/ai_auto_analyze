#!/bin/bash


source utils.sh
source profile_config.sh

create_clean_dir ${TRT_DOCKER_RESULTS_DIR}
write_run_metadata ${TRT_DOCKER_RESULTS_DIR} ${TRT_DOCKER_IMAGE}

log_info "Run profiles:"

for p in "${PROFILES[@]}"; do
    declare -n profile="$p"
    model=${profile[model]}
    gpu_ids=${profile[gpu_ids]}
    input_len=${profile[input_len]}
    output_len=${profile[output_len]}
    log_info "  Profiling:"
    log_info "    model      = ${model}"
    log_info "    gpu_ids    = ${gpu_ids}"
    log_info "    input_len  = ${input_len}"
    log_info "    output_len = ${output_len}"

    num_gpus=$(echo "${gpu_ids}" | awk -F',' '{print NF}')
    model_dirname=$(echo "${model}" | sed 's/\//__/g')

    model_dir=${TRT_DOCKER_RESULTS_DIR}/${model_dirname}
    create_dir_if_missing ${model_dir}

    datasets_dir=${model_dir}/${DATASETS_DIR}
    create_dir_if_missing ${datasets_dir}

    # Run prepare dataset
    log_info "Prepare TRT dataset for:"
    log_info "  MODEL      = ${model}"
    log_info "  INPUT_LEN  = ${input_len}"
    log_info "  OUTPUT_LEN = ${output_len}"
    datasets_file="${datasets_dir}/rand_dataset_isl_${input_len}_osl_${output_len}.txt"

    python3 /app/tensorrt_llm/benchmarks/cpp/prepare_dataset.py \
        --tokenizer=${model} \
        --stdout token-norm-dist \
        --num-requests=1000 \
        --input-mean=${input_len} \
        --output-mean=${output_len} \
        --input-stdev=0 \
        --output-stdev=0 > ${datasets_file}

    for concurrency in ${PROFILE_CONCURRENCIES}; do
        log_info "      Run concurrency = ${concurrency}"
        
        ((num_requests = concurrency * NUM_WAVES))
        
        # Set extra options
        mode=""
        yaml_flag=""
        if [[ -v profile[trt_mode] && -n "${profile[trt_mode]}" ]]; then
            mode="${profile[trt_mode]}"
            log_info "Set TRT MODE = ${mode}"
            yaml_file="trt/trt_mode_${mode}.yaml"
            log_info "Add yaml file: ${yaml_file}"
            yaml_flag="--extra_llm_api_options ${yaml_file}"
        fi
        
        # Generate test ID
        test_name="$(
            make_test_name \
                ${TRT} \
                ${model_dirname} \
                ${num_gpus} \
                ${concurrency} \
                ${input_len} \
                ${output_len} \
                ${mode}
            )"
        
        # Create test dir (for outputs)
        test_dir="$(
            make_test_dirname \
                ${model_dir} \
                ${test_name}
            )"
        create_dir_if_missing ${test_dir}
        
        # Create output filenames
        result_filename="$(
            make_result_filename \
                ${test_dir} \
                ${test_name}
            )"
        
        run_log_filename="$(
            make_run_log_filename \
                ${test_dir} \
                ${test_name}
            )"
        
        run_log_profile_filename="$(
            make_run_log_profile_filename \
                ${test_dir} \
                ${test_name}
            )"
        
        # Set profile env vars
        profile_prefix=""
        if is_trt_profile_enabled; then
            log_info "TRT profile is enabled."
            
            start_iter=$(calc_start_iter ${NUM_WARMUPS} ${NUM_WAVES} ${output_len})
            finish_iter=$(calc_finish_iter ${start_iter} ${NUM_TRACE_ITERS})

            export TLLM_PROFILE_START_STOP="${start_iter}-${finish_iter}"
            log_info "    Set TLLM_PROFILE_START_STOP=${TLLM_PROFILE_START_STOP}"

            trace_file_prefix="$(
                make_trace_file_prefix \
                    ${test_dir} \
                    ${test_name}
                )"

            profile_prefix="nsys profile ${NSYS_DEFAULT_FLAGS} -o ${trace_file_prefix}"
        fi

        # Run
        run_cmd="trtllm-bench \
            --model ${model} \
            throughput \
            --dataset ${datasets_file} \
            --num_requests ${num_requests} \
            --concurrency ${concurrency} \
            --tp ${num_gpus} \
            --eos_id -1 \
            --report_json ${result_filename} \
            --streaming \
            --warmup $NUM_WARMUPS \
            ${yaml_flag}"
        
        log_info "RUN NORMAL"
        run_and_log ${run_log_filename} \
            env CUDA_VISIBLE_DEVICES=${gpu_ids} \
            ${run_cmd}
            
        if is_vllm_profile_enabled; then
            log_info "RUN PROFILE"
            run_and_log ${run_log_profile_filename} \
                env CUDA_VISIBLE_DEVICES=${gpu_ids} \
                ${profile_prefix} ${run_cmd} 

            # Convert nsys binary file to SQLite format (python can read it)
            nsys export \
                --type=sqlite \
                --output="${trace_file_prefix}.sqlite" \
                ${trace_file_prefix}.nsys-rep
        fi
    done
done


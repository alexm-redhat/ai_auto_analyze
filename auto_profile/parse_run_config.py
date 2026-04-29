import json
import os
import sys

VALID_FRAMEWORKS = ("vllm", "sgl", "trt")
VALID_MODE_PREFIXES = tuple(f"{fw}_" for fw in VALID_FRAMEWORKS)

PROFILE_REQUIRED_FIELDS = {
    "model": str,
    "gpu_ids": (str, list),
}


def parse_modes(modes_list, profile_idx, errors):
    parsed = {}
    if not isinstance(modes_list, list):
        errors.append(f'"profiles[{profile_idx}].modes" must be a list.')
        return parsed

    for entry in modes_list:
        if not isinstance(entry, str):
            errors.append(
                f'"profiles[{profile_idx}].modes" entries must be strings. '
                f"Got: {entry!r}."
            )
            continue

        matched = False
        for fw in VALID_FRAMEWORKS:
            prefix = f"{fw}_"
            if entry.startswith(prefix):
                mode_name = entry[len(prefix):]
                if not mode_name:
                    errors.append(
                        f'"profiles[{profile_idx}].modes" entry "{entry}" '
                        f"has empty mode name after prefix."
                    )
                elif fw in parsed:
                    errors.append(
                        f'"profiles[{profile_idx}].modes" has duplicate '
                        f'framework prefix "{fw}": "{parsed[fw]}" and "{mode_name}".'
                    )
                else:
                    parsed[fw] = mode_name
                matched = True
                break

        if not matched:
            errors.append(
                f'"profiles[{profile_idx}].modes" entry "{entry}" must start '
                f"with a framework prefix: {', '.join(f'{fw}_' for fw in VALID_FRAMEWORKS)}."
            )

    return parsed


def validate_gpu_configs(gpu_configs, errors):
    if not isinstance(gpu_configs, dict):
        errors.append('"gpu_configs" must be an object mapping names to GPU ID lists.')
        return
    for name, ids in gpu_configs.items():
        if not isinstance(ids, list) or len(ids) == 0:
            errors.append(f'"gpu_configs.{name}" must be a non-empty list of GPU numbers.')
        elif not all(isinstance(g, int) and g >= 0 for g in ids):
            errors.append(f'"gpu_configs.{name}" must contain non-negative integers.')


def load_json_file(field_name, file_path, errors):
    if not os.path.isfile(file_path):
        errors.append(f'"{field_name}" path not found: {file_path}')
        return None

    with open(file_path) as f:
        return json.load(f)


def validate_docker_images(docker_images, errors):
    if not isinstance(docker_images, dict):
        errors.append('"docker_images_file" must contain an object mapping framework names to image strings.')
        return
    for name, image in docker_images.items():
        if name not in VALID_FRAMEWORKS:
            errors.append(
                f'docker_images contains invalid key "{name}". '
                f"Valid keys are: {list(VALID_FRAMEWORKS)}."
            )
        elif not isinstance(image, str) or len(image) == 0:
            errors.append(f'docker_images.{name} must be a non-empty string.')


def load_docker_images(docker_images_file, errors):
    docker_images = load_json_file("docker_images_file", docker_images_file, errors)
    if docker_images is not None:
        validate_docker_images(docker_images, errors)
    return docker_images


def validate_exec_modes(exec_modes, errors):
    if not isinstance(exec_modes, dict):
        errors.append('"exec_modes_file" must contain an object mapping names to {input_len, output_len}.')
        return
    for name, cfg in exec_modes.items():
        if not isinstance(cfg, dict):
            errors.append(f'exec_modes.{name} must be an object with "input_len" and "output_len".')
            continue
        if "input_len" not in cfg or not isinstance(cfg["input_len"], int) or cfg["input_len"] <= 0:
            errors.append(f'exec_modes.{name}.input_len must be a positive integer.')
        if "output_len" not in cfg or not isinstance(cfg["output_len"], int) or cfg["output_len"] <= 0:
            errors.append(f'exec_modes.{name}.output_len must be a positive integer.')


def resolve_exec_mode(i, p, exec_modes, errors):
    exec_mode = p.get("exec_mode")
    if exec_mode is None:
        return
    if not isinstance(exec_mode, str):
        errors.append(f'"profiles[{i}].exec_mode" must be a string.')
        return
    if exec_modes is None:
        errors.append(
            f'"profiles[{i}].exec_mode" references "{exec_mode}" '
            f'but no "exec_modes_file" is specified.'
        )
        return
    if exec_mode not in exec_modes:
        errors.append(
            f'"profiles[{i}].exec_mode" references "{exec_mode}" '
            f"which is not defined in exec_modes. "
            f"Available: {list(exec_modes.keys())}."
        )
        return
    cfg = exec_modes[exec_mode]
    p["input_len"] = cfg["input_len"]
    p["output_len"] = cfg["output_len"]


def load_gpu_configs(gpu_configs_file, errors):
    gpu_configs = load_json_file("gpu_configs_file", gpu_configs_file, errors)
    if gpu_configs is not None:
        validate_gpu_configs(gpu_configs, errors)
    return gpu_configs


def load_exec_modes(exec_modes_file, errors):
    exec_modes = load_json_file("exec_modes_file", exec_modes_file, errors)
    if exec_modes is not None:
        validate_exec_modes(exec_modes, errors)
    return exec_modes


def resolve_gpu_ids(i, p, gpu_configs, errors):
    gpu_ids = p.get("gpu_ids")
    if gpu_ids is None:
        return
    if isinstance(gpu_ids, str):
        if gpu_configs is None:
            errors.append(
                f'"profiles[{i}].gpu_ids" references "{gpu_ids}" '
                f'but no "gpu_configs_file" is specified.'
            )
        elif gpu_ids not in gpu_configs:
            errors.append(
                f'"profiles[{i}].gpu_ids" references "{gpu_ids}" '
                f"which is not defined in gpu_configs. "
                f"Available: {list(gpu_configs.keys())}."
            )
        else:
            p["gpu_ids"] = gpu_configs[gpu_ids]


def validate_profile(i, p, errors):
    if not isinstance(p, dict):
        errors.append(
            f'"profiles[{i}]" must be an object with fields: '
            f'{", ".join(PROFILE_REQUIRED_FIELDS)}.'
        )
        return

    for field, ftype in PROFILE_REQUIRED_FIELDS.items():
        if field not in p:
            errors.append(f'"profiles[{i}].{field}" is required.')
        elif not isinstance(p[field], ftype):
            errors.append(f'"profiles[{i}].{field}" must be of type {ftype.__name__}.')

    if "model" in p and isinstance(p["model"], str) and len(p["model"]) == 0:
        errors.append(f'"profiles[{i}].model" must be a non-empty string.')

    if "gpu_ids" in p and isinstance(p["gpu_ids"], list):
        if len(p["gpu_ids"]) == 0:
            errors.append(f'"profiles[{i}].gpu_ids" must be a non-empty list of GPU numbers.')
        elif not all(isinstance(g, int) and g >= 0 for g in p["gpu_ids"]):
            errors.append(f'"profiles[{i}].gpu_ids" must contain non-negative integers.')

    has_exec_mode = "exec_mode" in p
    has_inline = "input_len" in p or "output_len" in p
    if not has_exec_mode and not has_inline:
        errors.append(
            f'"profiles[{i}]" must specify either "exec_mode" '
            f'or both "input_len" and "output_len".'
        )
    elif not has_exec_mode:
        if "input_len" not in p:
            errors.append(f'"profiles[{i}].input_len" is required when "exec_mode" is not set.')
        elif not isinstance(p["input_len"], int) or p["input_len"] <= 0:
            errors.append(f'"profiles[{i}].input_len" must be a positive integer.')
        if "output_len" not in p:
            errors.append(f'"profiles[{i}].output_len" is required when "exec_mode" is not set.')
        elif not isinstance(p["output_len"], int) or p["output_len"] <= 0:
            errors.append(f'"profiles[{i}].output_len" must be a positive integer.')

    if "modes" in p:
        parse_modes(p["modes"], i, errors)


def shell_escape(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def load_infra_config(infra_path, errors):
    infra = load_json_file("infra_config", infra_path, errors)
    if infra is None:
        return None, None, None

    if not isinstance(infra, dict):
        errors.append("infra_config must be a JSON object.")
        return None, None, None

    docker_images = None
    docker_images_file = infra.get("docker_images_file")
    if docker_images_file is not None:
        if not isinstance(docker_images_file, str):
            errors.append('"docker_images_file" must be a string path.')
        else:
            docker_images = load_docker_images(docker_images_file, errors)

    gpu_configs = None
    gpu_configs_file = infra.get("gpu_configs_file")
    if gpu_configs_file is not None:
        if not isinstance(gpu_configs_file, str):
            errors.append('"gpu_configs_file" must be a string path.')
        else:
            gpu_configs = load_gpu_configs(gpu_configs_file, errors)

    exec_modes = None
    exec_modes_file = infra.get("exec_modes_file")
    if exec_modes_file is not None:
        if not isinstance(exec_modes_file, str):
            errors.append('"exec_modes_file" must be a string path.')
        else:
            exec_modes = load_exec_modes(exec_modes_file, errors)

    return docker_images, gpu_configs, exec_modes


def parse_config(infra_config_path, run_config_path):
    with open(run_config_path) as f:
        c = json.load(f)

    errors = []

    docker_images, gpu_configs, exec_modes = load_infra_config(
        infra_config_path, errors
    )

    profiles = c.get("profiles", [])
    if not isinstance(profiles, list) or len(profiles) == 0:
        errors.append('"profiles" must be a non-empty list.')
    else:
        for i, p in enumerate(profiles):
            resolve_gpu_ids(i, p, gpu_configs, errors)
            resolve_exec_mode(i, p, exec_modes, errors)
            validate_profile(i, p, errors)

    if "concurrencies" not in c or not isinstance(c["concurrencies"], list) or len(c["concurrencies"]) == 0:
        errors.append('"concurrencies" must be a non-empty list of integers.')

    frameworks = c.get("frameworks", [])
    if not isinstance(frameworks, list):
        errors.append('"frameworks" must be a list.')
        frameworks = []
    invalid_fw = [f for f in frameworks if f not in VALID_FRAMEWORKS]
    if invalid_fw:
        errors.append(
            f'"frameworks" contains invalid entries: {invalid_fw}. '
            f"Valid options are: {list(VALID_FRAMEWORKS)}."
        )

    enable_traces = c.get("enable_traces", [])
    if not isinstance(enable_traces, list):
        errors.append('"enable_traces" must be a list.')
        enable_traces = []
    invalid_tr = [t for t in enable_traces if t not in VALID_FRAMEWORKS]
    if invalid_tr:
        errors.append(
            f'"enable_traces" contains invalid entries: {invalid_tr}. '
            f"Valid options are: {list(VALID_FRAMEWORKS)}."
        )

    traces_not_in_fw = [t for t in enable_traces if t in VALID_FRAMEWORKS and t not in frameworks]
    if traces_not_in_fw:
        errors.append(
            f'"enable_traces" lists frameworks not in "frameworks": {traces_not_in_fw}. '
            f'A framework must be in "frameworks" to enable tracing for it.'
        )

    if docker_images is not None and frameworks:
        fw_missing_image = [fw for fw in frameworks if fw not in docker_images]
        if fw_missing_image:
            errors.append(
                f'"frameworks" lists {fw_missing_image} but no docker image is defined '
                f"for them in docker_images_file."
            )
    elif docker_images is None and frameworks:
        errors.append(
            '"frameworks" is non-empty but no "docker_images_file" is specified in infra config.'
        )

    if errors:
        for e in errors:
            print(f'log_error "Config error: {e}"')
        print("return 1")
        return

    parsed_modes = []
    for p in profiles:
        parsed_modes.append(parse_modes(p.get("modes", []), 0, []))

    concurrencies = " ".join(str(x) for x in c["concurrencies"])

    def make_assoc(value_fn):
        entries = " ".join(
            f'["{shell_escape(p["model"])}"]="{shell_escape(value_fn(p, pm))}"'
            for p, pm in zip(profiles, parsed_modes)
        )
        return entries

    profiles_arr = " ".join(f'"{shell_escape(p["model"])}"' for p in profiles)

    for fw in VALID_FRAMEWORKS:
        var_name = f"{fw.upper()}_DOCKER_IMAGE"
        image = docker_images.get(fw, "")
        print(f'{var_name}="{shell_escape(image)}"')

    print(f"PROFILES=({profiles_arr})")
    print(f"declare -gA PROFILE_GPU_IDS=({make_assoc(lambda p, _: ','.join(str(g) for g in p['gpu_ids']))})")
    print(f"declare -gA PROFILE_INPUT_LENS=({make_assoc(lambda p, _: str(p['input_len']))})")
    print(f"declare -gA PROFILE_OUTPUT_LENS=({make_assoc(lambda p, _: str(p['output_len']))})")
    print(f"declare -gA PROFILE_VLLM_MODES=({make_assoc(lambda _, pm: pm.get('vllm', 'none'))})")
    print(f"declare -gA PROFILE_SGL_MODES=({make_assoc(lambda _, pm: pm.get('sgl', 'none'))})")
    print(f"declare -gA PROFILE_TRT_MODES=({make_assoc(lambda _, pm: pm.get('trt', 'none'))})")
    print(f'PROFILE_CONCURRENCIES="{concurrencies}"')
    print(f'RUN_FRAMEWORKS="{" ".join(frameworks)}"')
    print(f'ENABLE_TRACES="{" ".join(enable_traces)}"')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('log_error "Usage: parse_run_config.py <infra_config.json> <run_config.json>"')
        print("return 1")
        sys.exit(1)
    parse_config(sys.argv[1], sys.argv[2])

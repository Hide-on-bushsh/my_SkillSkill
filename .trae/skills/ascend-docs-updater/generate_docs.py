"""
Parse test files from performance directory and generate best practice documentation.
"""
import os
import re
import sys
import ast

PERFORMANCE_DIR = r"c:\Users\xujianzhao\Desktop\Ascend\sglang\test\registered\ascend\performance"
OUTPUT_DIR = r"c:\Users\xujianzhao\Desktop\sglang\docs_new\docs\hardware-platforms\ascend-npus\best_practice"

MODEL_DISPLAY_NAMES = {
    "deepseek_r1": "DeepSeek-R1",
    "deepseek_v3_2": "DeepSeek-V3.2",
    "deepseek_v4_flash": "DeepSeek-V4-Flash",
    "glm5_1": "GLM-5.1",
    "kimi_k2_6": "Kimi-K2.6",
    "mimo_v2_flash": "MiMo-V2-Flash",
    "minimax_m2_5": "MiniMax-M2.5",
    "qwen3-8b": "Qwen3-8B",
    "qwen3_235b_a22b": "Qwen3-235B-A22B",
    "qwen3_30b_a3b": "Qwen3-30B-A3B",
    "qwen3_32b": "Qwen3-32B",
    "qwen3_5_397b": "Qwen3.5-397B",
    "qwen3_6_27b": "Qwen3.6-27B",
    "qwen3_6_35b_a3b": "Qwen3.6-35B-A3B",
    "qwen3_next_80b_a3b_instruct": "Qwen3-Next-80B-A3B-Instruct",
}

# Slug override for model tutorial pages whose filename differs from model_dir.
# Keys are model_dir names; values are the corresponding tutorial slug (filename
# without extension) under docs/hardware-platforms/ascend-npus/model-tutorials/.
MODEL_TUTORIAL_SLUG_OVERRIDES = {
    "glm5_1": "glm_5_1",
}


def get_tutorial_slug(model_dir):
    """Return the model-tutorials slug for a given model_dir."""
    return MODEL_TUTORIAL_SLUG_OVERRIDES.get(model_dir, model_dir)


# CLI args to exclude from generated documentation (e.g. internal paths, debug flags).
EXCLUDED_ARGS = {
    "--init-expert-location",
}

# Env vars to exclude from generated documentation (development / debugging only).
EXCLUDED_ENV_VARS = {
    "SGLANG_NPU_PROFILING",
    "SGLANG_NPU_PROFILING_STAGE",
    "SGLANG_NPU_PROFILING_BS",
    "SGLANG_PROFILE_WITH_STACK",
}


def get_hardware(filename):
    if "a2" in filename.lower():
        return "Atlas 800I A2"
    return "Atlas 800I A3"


def parse_quantization(filename):
    name = filename.lower()
    if "w4a8" in name:
        return "W4A8 INT8"
    elif "w8a8" in name:
        return "W8A8 INT8"
    elif "bf16" in name:
        return "BF16"
    return "BF16"


def parse_deploy_mode(filename, is_multi_node):
    name = filename.lower()
    if "pd_sep" in name or "1p1d" in name or "2p1d" in name:
        return "PD Disaggregation"
    if is_multi_node:
        return "PD Disaggregation"
    return "PD Mixed"


def safe_val(val):
    if isinstance(val, (int, float)):
        return str(int(val)) if val == int(val) else str(val)
    return str(val).strip()


def parse_dataset_from_filename(filename):
    """Extract dataset string from filename like in128k_out1k → 128k+1k, with optional prefix suffix."""
    m = re.search(r'_in([\d.kpqx_]+)_out([\d.k]+)(?:[_\.]|$)', filename.lower())
    if m:
        inp = m.group(1).rstrip("_")
        out = m.group(2).rstrip("_")
        suffix = ""
        after = filename.lower()[m.end():]
        prefix_m = re.match(r'prefix(\d+)', after)
        if prefix_m:
            suffix = f" ({prefix_m.group(1)}% prefix cache hit rate)"
        def fmt(s):
            if "x" in s.lower():
                m2 = re.match(r'^(\d+x\d+)_?(\d+)$', s)
                if m2:
                    return f"{m2.group(1)} ({m2.group(2)})"
                return s
            if "k" in s.lower():
                s = re.sub(r'^(\d+)k(\d+)', r'\1.\2k', s)
                if s.endswith("k"):
                    return s
                return s[:-1] + "k"
            return s
        return f"{fmt(inp)}+{fmt(out)}{suffix}"
    return ""


def parse_python_list_value(source_text):
    """Try to parse a Python expression like a dict or list using AST."""
    try:
        node = ast.parse(source_text.strip(), mode='eval')
        return node.body
    except Exception:
        return None


def extract_python_dict(source, var_name):
    """Extract a Python dictionary value for a given variable name using AST."""
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if isinstance(node.value, ast.Dict):
                            result = {}
                            for k, v in zip(node.value.keys, node.value.values):
                                key = k.value if isinstance(k, ast.Constant) else None
                                val = _node_to_str(v)
                                if val is not None:
                                    result[key] = val
                            return result
    except Exception:
        pass
    return None


def _node_to_str(node):
    """Convert an AST node to its string representation."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            return str(-node.operand.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return ast.unparse(node) if hasattr(ast, 'unparse') else None
    if isinstance(node, ast.JoinedStr):
        try:
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                elif isinstance(v, ast.FormattedValue):
                    val = _node_to_str(v.value)
                    parts.append(f"{{{val}}}" if val else "")
            return "".join(parts)
        except Exception:
            return None
    return None


def extract_python_list(source, var_name):
    """Extract a Python list value for a given variable name using AST."""
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        if isinstance(node.value, ast.List):
                            result = []
                            for elt in node.value.elts:
                                result.append(_node_to_str(elt))
                            return result
    except Exception:
        pass
    return None


def extract_model_config(source, var_name=None):
    """Extract MODEL_CONFIG using AST for multi-node tests.
    Also handles indirect references (PREFILL_ENVS, PREFILL_ARGS, etc.).
    If var_name is given, searches for that specific variable. Otherwise searches
    for any variable ending with _MODEL_CONFIG."""
    try:
        tree = ast.parse(source)
        # First pass: collect all top-level variable assignments
        top_level = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Dict):
                            inner = {}
                            for ik, iv in zip(node.value.keys, node.value.values):
                                ik_val = ik.value if isinstance(ik, ast.Constant) else None
                                val = _node_to_str(iv)
                                if val is not None:
                                    inner[ik_val] = val
                            top_level[target.id] = inner
                        elif isinstance(node.value, ast.List):
                            inner = []
                            for elt in node.value.elts:
                                val = _node_to_str(elt)
                                inner.append(val)
                            top_level[target.id] = inner
                        elif isinstance(node.value, ast.Constant):
                            top_level[target.id] = str(node.value.value)

        # Second pass: find MODEL_CONFIG and resolve references
        def _resolve_config(name):
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            if isinstance(node.value, ast.Dict):
                                result = {}
                                for k, v in zip(node.value.keys, node.value.values):
                                    key = k.value if isinstance(k, ast.Constant) else None
                                    if isinstance(v, ast.Constant):
                                        result[key] = str(v.value)
                                    elif isinstance(v, ast.Dict):
                                        inner = {}
                                        for ik, iv in zip(v.keys, v.values):
                                            ik_val = ik.value if isinstance(ik, ast.Constant) else None
                                            val = _node_to_str(iv)
                                            if val is not None:
                                                inner[ik_val] = val
                                        result[key] = inner
                                    elif isinstance(v, ast.List):
                                        inner = []
                                        for elt in v.elts:
                                            val = _node_to_str(elt)
                                            inner.append(val)
                                        result[key] = inner
                                    elif isinstance(v, ast.Name):
                                        ref_name = v.id
                                        if ref_name in top_level and top_level[ref_name] is not None:
                                            result[key] = top_level[ref_name]
                                return result
            return None

        if var_name:
            return _resolve_config(var_name)

        # Search for any *_MODEL_CONFIG or MODEL_CONFIG variable
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and (target.id == "MODEL_CONFIG" or target.id.endswith("_MODEL_CONFIG")):
                        result = _resolve_config(target.id)
                        if result:
                            return result
    except Exception:
        pass

    # Fallback to regex for simple cases
    result = {}
    for section in ["prefill_envs", "decode_envs", "router_envs"]:
        m = re.search(rf'"{section}"\s*:\s*\{{(.*?)\}}', source, re.DOTALL)
        if m:
            envs = {}
            for line in m.group(1).strip().split("\n"):
                line = line.strip().rstrip(",")
                kv = re.match(r'"(\w+)":\s*"([^"]*)"', line)
                if kv:
                    envs[kv.group(1)] = kv.group(2)
            result[section] = envs

    for section in ["prefill_args", "decode_args", "router_args"]:
        m = re.search(rf'"{section}"\s*:\s*\[(.*?)\]', source, re.DOTALL)
        if m:
            args = []
            content = m.group(1)
            # Find all quoted strings in the list
            for token in re.finditer(r'"([^"]*)"', content):
                args.append(token.group(1))
            result[section] = args
    return result


def extract_config_from_file(filepath):
    """Extract all configuration from a test file."""
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    config = {
        "is_multi_node": False,
        "is_pd_separate": False,
        "envs": {},
        "other_args": [],
        "prefill_envs": {},
        "decode_envs": {},
        "router_envs": {},
        "prefill_args": [],
        "decode_args": [],
        "router_args": [],
        "benchmark": {},
    }

    filename = os.path.basename(filepath)
    config["filename"] = filename
    config["hardware"] = get_hardware(filename)
    config["quantization"] = parse_quantization(filename)

    # Check if multi-node - look for TestAscendPerfMultiNodePdSepTestCaseBase or PdMix
    is_sep = "TestAscendPerfMultiNodePdSepTestCaseBase" in source
    is_mix = "TestAscendPerfMultiNodePdMixTestCaseBase" in source
    is_multi = is_sep or is_mix
    config["is_multi_node"] = is_multi
    config["is_pd_separate"] = is_sep

    config["deploy_mode"] = parse_deploy_mode(filename, is_sep)

    # Extract MODEL_CONFIG for multi-node
    if is_multi:
        mc = extract_model_config(source)
        if mc:
            if is_sep:
                config["prefill_envs"] = mc.get("prefill_envs", {})
                config["decode_envs"] = mc.get("decode_envs", {})
                config["router_envs"] = mc.get("router_envs", {})
                config["prefill_args"] = [a for a in mc.get("prefill_args", []) if a is not None]
                config["decode_args"] = [a for a in mc.get("decode_args", []) if a is not None]
                config["router_args"] = [a for a in mc.get("router_args", []) if a is not None]
            else:
                # PdMix: all nodes share same envs/args
                config["envs"] = mc.get("node_envs", {})
                config["other_args"] = [a for a in mc.get("other_args", []) if a is not None]

            # Cards are always from filename (e.g. _16p_ = 16 cards)
            config["cards"] = parse_cards_from_filename(filename)
    else:
        # Find envs variable: any variable ending with _ENVS or exactly ENVS assigned a dict
        envs_var_names = re.findall(r'^(\w*_?ENVS)\s*=\s*\{', source, re.MULTILINE)
        if not envs_var_names:
            envs_var_names = re.findall(r'^(ENVS)\s*=\s*\{', source, re.MULTILINE)
        for var_name in envs_var_names:
            envs = extract_python_dict(source, var_name)
            if envs:
                config["envs"] = envs
                break

        # Find args variable: any variable ending with _OTHER_ARGS, _ARGS, or exactly ARGS
        args_var_names = re.findall(r'^(\w*(?:_OTHER_ARGS|_ARGS))\s*=\s*\[', source, re.MULTILINE)
        if not args_var_names:
            args_var_names = re.findall(r'^(OTHER_ARGS|ARGS)\s*=\s*\[', source, re.MULTILINE)
        for var_name in args_var_names:
            args = extract_python_list(source, var_name)
            if args:
                config["other_args"] = [a for a in args if a is not None]
                break

        # Determine cards from filename
        config["cards"] = parse_cards_from_filename(filename)

    # Extract benchmark from performance class (skip accuracy classes)
    for class_match in re.finditer(
        r'class\s+(\w+)\((.+?)\):(.*?)(?=\nclass\s+|\nif\s+__name__)', source, re.DOTALL
    ):
        class_name = class_match.group(1)
        class_base = class_match.group(2)
        class_body = class_match.group(3)

        # Skip accuracy test classes by name or base class
        if ("accuracy" in class_name.lower() or "mmlu" in class_name.lower() or
            "gpqa" in class_name.lower() or "aime" in class_name.lower() or
            "Accuracy" in class_base or "accuracy" in class_base):
            continue

        bm = config["benchmark"]

        # Strip docstrings to avoid matching content inside them
        class_body_clean = re.sub(r'""".*?"""', '', class_body, flags=re.DOTALL)

        # Dynamically extract all ``identifier = value`` assignments from the
        # class body (instead of a hardcoded list).  Only skip known config
        # plumbing fields that are never benchmark parameters.
        _non_bm_fields = {"model_config", "benchmark_tool", "dataset_type",
                          "model", "model_path", "other_args", "envs",
                          "other_envs", "model_type"}
        for m in re.finditer(r'^\s*(\w+)\s*=\s*([^\n]+)', class_body_clean, re.MULTILINE):
            field = m.group(1)
            if field in _non_bm_fields:
                continue
            val = m.group(2).strip()
            # Strip surrounding quotes
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            try:
                if "." in val and re.match(r'^-?[\d.]+$', val):
                    bm[field] = float(val)
                elif val.isdigit():
                    bm[field] = int(val)
                else:
                    bm[field] = val
            except:
                bm[field] = val

        # Use mean_e2e_latency as tpot fallback
        if "mean_e2e_latency" in bm and "tpot" not in bm:
            bm["tpot"] = bm["mean_e2e_latency"]

        # Resolve expressions in benchmark values
        if "num_prompts" in bm and isinstance(bm["num_prompts"], str):
            if "int(max_concurrency)" in bm["num_prompts"] and "max_concurrency" in bm:
                m = re.search(r'\*\s*(\d+)', bm["num_prompts"])
                if m:
                    bm["num_prompts"] = int(bm["max_concurrency"]) * int(m.group(1))
        if "request_rate" in bm and isinstance(bm["request_rate"], str):
            if 'inf' in bm["request_rate"]:
                bm["request_rate"] = "inf"

        # Found a performance class, stop searching
        break

    # Determine is_multi_node based on actual --nnodes value, not class inheritance
    def _get_nnodes(args_list):
        for i, a in enumerate(args_list):
            if isinstance(a, str) and a == "--nnodes" and i + 1 < len(args_list):
                val = args_list[i + 1]
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
        return 1

    if config["is_pd_separate"]:
        pf_nn = _get_nnodes(config.get("prefill_args", []))
        dc_nn = _get_nnodes(config.get("decode_args", []))
        config["is_multi_node"] = (pf_nn > 1 or dc_nn > 1)
    else:
        nn = _get_nnodes(config.get("other_args", []))
        config["is_multi_node"] = (nn > 1)

    # Override quantization: if no --quantization in args, it's BF16.
    all_args = (config.get("prefill_args", []) + config.get("decode_args", []) +
                config.get("other_args", []))
    if "--quantization" not in all_args:
        config["quantization"] = "BF16"

    # Cross-validate with model path constant name and resolved path.
    mp_quant = _parse_quant_from_model_path(source)
    if mp_quant and config["quantization"].upper() != mp_quant.upper():
        config["quantization"] = mp_quant

    return config


_model_path_map = None


def _get_model_path_map():
    """Read test_npu_performance_utils.py, return {VARNAME: path_string} for *_MODEL_PATH."""
    global _model_path_map
    if _model_path_map is not None:
        return _model_path_map

    utils_path = os.path.normpath(os.path.join(
        os.path.dirname(PERFORMANCE_DIR),
        "..", "..", "..", "python", "sglang", "test", "ascend", "e2e",
        "test_npu_performance_utils.py",
    ))
    _model_path_map = {}
    try:
        with open(utils_path, "r", encoding="utf-8") as f:
            utils_source = f.read()
        tree = ast.parse(utils_source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_MODEL_PATH"):
                        _model_path_map[target.id] = _node_to_str(node.value)
    except Exception:
        pass
    return _model_path_map


def _parse_quant_from_model_path(source):
    """Extract quantization (W4A8, W8A8, BF16) from model path info."""
    # 1. Variable name: DEEPSEEK_R1_W4A8_PER_CHANNEL_MODEL_PATH
    m = re.search(r'(\w+_(?:W4A8|W8A8|BF16)\w*)_MODEL_PATH\b', source)
    if m:
        token = m.group(1)
        if "W4A8" in token: return "W4A8 INT8"
        if "W8A8" in token: return "W8A8 INT8"
        if "BF16" in token: return "BF16"

    # 2. Resolved path string from utils: /.../MiMo-V2-Flash-W8A8/
    for var_name, path_str in _get_model_path_map().items():
        if var_name in source and path_str:
            for q, label in [("W8A8", "W8A8 INT8"), ("W4A8", "W4A8 INT8"), ("BF16", "BF16")]:
                if re.search(rf'[_-]{q}[/"\'\\]|{q}$', str(path_str), re.IGNORECASE):
                    return label
            break
    return None


def parse_pd_node_counts(filename):
    """Parse prefill and decode node counts from filename patterns like 2p1d, 1p1d."""
    name = filename.lower().replace(".py", "")
    m = re.search(r'_(\d+)p(\d*)d_', name)
    if m:
        prefill_count = int(m.group(1))
        decode_count = int(m.group(2)) if m.group(2) else 1
        return prefill_count, decode_count
    return 1, 1


def parse_cards_from_filename(filename):
    """Parse number of cards from filename."""
    name = filename.lower()
    for part in name.split("_"):
        if part.endswith("p") and not part.endswith("dp"):
            num_part = part[:-1]
            if num_part.endswith("1d"):
                continue
            if num_part.isdigit():
                return int(num_part)
    return 1


def _env_value_needs_quoting(v):
    """Return True if an env value needs to be wrapped in quotes for bash.

    Values containing shell metacharacters (e.g. ``;``, spaces, ``<>``) would
    otherwise break the ``export KEY=value`` statement or trigger unintended
    shell behavior (command separation, redirection, globbing, ...).
    """
    if v == "":
        return False
    # Placeholder values like ``<network-interface>`` are meant to be replaced
    # by the user — quoting them is unnecessary noise.
    if v.startswith("<") and v.endswith(">"):
        return False
    # Safe characters that never need quoting in an export value.
    return bool(re.search(r'[^A-Za-z0-9_./:=,@%+-]', v))


def format_env_exports(envs):
    lines = []
    for k, v in sorted(envs.items()):
        if k in EXCLUDED_ENV_VARS:
            continue
        # Clean up f-string residuals and variable refs in values
        v = re.sub(r'\{[A-Z][A-Z0-9_]*\}', 'xxx', v)
        # Clean up trailing colon from collapsed f-strings
        v = re.sub(r':\s*$', '', v)
        if _env_value_needs_quoting(v):
            # Escape any embedded double quotes before wrapping.
            v = '"' + v.replace('"', '\\"') + '"'
        lines.append(f"export {k}={v}")
    return "\n".join(lines)


def _resolve_var_name(val):
    """Resolve well-known Python variable names to bash values."""
    if not val:
        return val
    if val == "ROUND_ROBIN":
        return "round_robin"
    if val.isupper() and ("_PATH" in val or "_MODEL" in val):
        return "$DRAFT_MODEL_PATH"
    return val

def _arg_value_needs_quoting(v):
    """Return True if a CLI argument value needs to be wrapped in quotes for bash.

    Values containing shell metacharacters (spaces, braces, quotes, ``;``, ``$``,
    etc.) would otherwise be split or interpreted by the shell.
    """
    if v == "":
        return False
    # Bash variable references (e.g. ``$DRAFT_MODEL_PATH``) must not be quoted
    # — single quotes would prevent expansion.
    if v.startswith("$"):
        return False
    return bool(re.search(r'[^A-Za-z0-9_./:=,@%+-]', v))


def _quote_arg_value(v):
    """Wrap a CLI argument value in single quotes for bash.

    Single quotes are preferred over double quotes for values that may contain
    JSON (which uses double quotes internally). Embedded single quotes are
    handled via the standard ``'\\''`` escape sequence.
    """
    return "'" + v.replace("'", "'\\''") + "'"


def format_args_for_bash(args, indent=""):
    """Format a list of args into bash command line arguments."""
    # Filter out None values
    args = [a for a in args if a is not None]
    # Resolve Python variable names in arguments
    parts = []
    seen_flags = set()
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            if arg in EXCLUDED_ARGS:
                # Skip the flag and its value.
                i += 1
                while i < len(args) and args[i] and not args[i].startswith("--"):
                    i += 1
                continue
            if arg in seen_flags:
                # Deduplicate repeated flags — keep the first occurrence.
                i += 1
                while i < len(args) and args[i] and not args[i].startswith("--"):
                    i += 1
                continue
            seen_flags.add(arg)
            flag_parts = [arg]
            i += 1
            while i < len(args) and args[i] and not args[i].startswith("--"):
                val = _resolve_var_name(args[i])
                if _arg_value_needs_quoting(val):
                    val = _quote_arg_value(val)
                flag_parts.append(val)
                i += 1
            parts.append(" ".join(flag_parts))
        else:
            i += 1

    if not parts:
        return ""

    result = parts[0]
    for p in parts[1:]:
        result += f" \\\n{indent}{p}"
    return result


def _is_nic_var(v):
    """Check if a value is a NIC name variable (not a real NIC name)."""
    return v and v not in ("lo", "bond", "") and v.isupper()


def _filter_envs(envs, is_prefill=True):
    """Filter env vars: always use <network-interface> placeholder."""
    result = {}
    for k, v in sorted(envs.items()):
        if k in ("HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
            result[k] = "<network-interface>"
        else:
            result[k] = v
    return result


def format_pd_separate_command(config):
    prefill_envs = config.get("prefill_envs", {})
    decode_envs = config.get("decode_envs", {})
    router_envs = config.get("router_envs", {})
    prefill_args = config.get("prefill_args", [])
    decode_args = config.get("decode_args", [])
    router_args = config.get("router_args", [])

    # Find common envs (same key + value in both prefill and decode)
    # HCCL/GLOO always go into per-section envs since they need context-appropriate values
    common_envs = {}
    for k in prefill_envs:
        if k in ("HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
            continue
        if k in decode_envs and prefill_envs[k] == decode_envs[k]:
            common_envs[k] = prefill_envs[k]
    # Remove common from prefill/decode so they only appear in common
    prefill_only = {k: v for k, v in prefill_envs.items() if k not in common_envs}
    decode_only = {k: v for k, v in decode_envs.items() if k not in common_envs}

    common_envs_filtered = _filter_envs(common_envs, is_prefill=True)
    has_pythonpath_pd = "PYTHONPATH" in common_envs_filtered
    if has_pythonpath_pd:
        del common_envs_filtered["PYTHONPATH"]
    prefill_envs_filtered = _filter_envs(prefill_only, is_prefill=True)
    decode_envs_filtered = _filter_envs(decode_only, is_prefill=False)

    # Remove --disaggregation-mode from args since we add it explicitly with --host/--port
    def strip_flag(args, flag):
        result = []
        skip = False
        for a in args:
            if skip:
                skip = False
                continue
            if a == flag:
                skip = True
                continue
            result.append(a)
        return result

    prefill_args = strip_flag(prefill_args, "--disaggregation-mode")
    decode_args = strip_flag(decode_args, "--disaggregation-mode")
    prefill_args = strip_flag(prefill_args, "--node-rank")
    decode_args = strip_flag(decode_args, "--node-rank")

    if "--disaggregation-transfer-backend" not in prefill_args:
        prefill_args.extend(["--disaggregation-transfer-backend", "ascend"])
    if "--disaggregation-transfer-backend" not in decode_args:
        decode_args.extend(["--disaggregation-transfer-backend", "ascend"])
    if "--trust-remote-code" not in prefill_args:
        prefill_args.append("--trust-remote-code")
    if "--trust-remote-code" not in decode_args:
        decode_args.append("--trust-remote-code")
    if "--attention-backend" not in prefill_args:
        prefill_args.extend(["--attention-backend", "ascend"])
    if "--attention-backend" not in decode_args:
        decode_args.extend(["--attention-backend", "ascend"])
    if "--device" not in prefill_args:
        prefill_args.extend(["--device", "npu"])
    if "--device" not in decode_args:
        decode_args.extend(["--device", "npu"])

    prefill_args_str = format_args_for_bash(prefill_args, indent="        ")
    decode_args_str = format_args_for_bash(decode_args, indent="        ")

    decode_nnodes = 1
    for i, arg in enumerate(decode_args):
        if arg == "--nnodes" and i + 1 < len(decode_args):
            try:
                decode_nnodes = int(decode_args[i + 1])
            except (ValueError, TypeError):
                pass

    # Parse deployment topology from filename and args
    # pd_prefill = independent prefill groups (2p1d → 2, 1p1d → 1)
    # prefill_nnodes = nodes within ONE prefill group
    # p_ip_count = pd_prefill * prefill_nnodes (total prefill IPs)
    # d_ip_count = decode_nnodes (decode is always 1 group)
    filename = config.get("filename", "")
    pd_prefill, _ = parse_pd_node_counts(filename)
    prefill_nnodes = 1
    had_nnodes = False
    for i, arg in enumerate(prefill_args):
        if arg == "--nnodes" and i + 1 < len(prefill_args):
            try:
                prefill_nnodes = int(prefill_args[i + 1])
                had_nnodes = True
            except (ValueError, TypeError):
                pass
    p_ip_count = pd_prefill * prefill_nnodes
    d_ip_count = decode_nnodes

    has_draft = ("--speculative-draft-model-path" in prefill_args or
                 "--speculative-draft-model-path" in decode_args)

    # Build variable comment header
    comment_items = [
        "#   P_IP: prefill node IP address",
        "#   D_IP: decode node IP address",
        "#   ASCEND_MF_STORE_URL: prefill node IP with port",
        "#   MODEL_PATH: path to the model weights directory",
    ]
    if has_draft:
        comment_items.append("#   DRAFT_MODEL_PATH: path to the draft model weights directory")
    comment_items += [
        "#   HCCL_SOCKET_IFNAME: network interface name for HCCL",
        "#   GLOO_SOCKET_IFNAME: network interface name for Gloo",
    ]
    comment_lines = [
        "# ============================================================",
        "# Before running, update the following variables:",
    ] + comment_items + [
        "# ============================================================",
    ]

    lines = [
        "",
        "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
        "sysctl -w vm.swappiness=0",
        "sysctl -w kernel.numa_balancing=0",
        "sysctl -w kernel.sched_migration_cost_ns=50000",
        "",
        "unset https_proxy",
        "unset http_proxy",
        "unset HTTPS_PROXY",
        "unset HTTP_PROXY",
        "unset ASCEND_LAUNCH_BLOCKING",
        "",
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh",
        "source /usr/local/Ascend/nnal/atb/set_env.sh",
        "",
    ]

    if common_envs_filtered:
        lines.append(format_env_exports(common_envs_filtered))
        lines.append("")

    if p_ip_count > 1:
        ips = " ".join(f"'<your prefill ip{i+1}>'" for i in range(p_ip_count))
        mf_store_ip = "<your prefill ip1>"
    else:
        ips = "'<your prefill ip>'"
        mf_store_ip = "<your prefill ip>"
    lines.append(f"P_IP=({ips})")

    if d_ip_count > 1:
        ips = " ".join(f"'<your decode ip{i+1}>'" for i in range(d_ip_count))
    else:
        ips = "'<your decode ip>'"
    lines.append(f"D_IP=({ips})")

    lines.append("")
    lines.append(f'export ASCEND_MF_STORE_URL="tcp://{mf_store_ip}:24670"')
    lines.append("")

    lines.append("MODEL_PATH=/path/to/model-weights")
    if has_draft:
        lines.append("DRAFT_MODEL_PATH=/path/to/draft-model-weights")
        if has_pythonpath_pd:
            lines.append("export PYTHONPATH=${DRAFT_MODEL_PATH}:$PYTHONPATH")

    lines.extend([
        "",
        'LOCAL_HOST1=`hostname -I|awk -F " " \'{print$1}\'`',
        'LOCAL_HOST2=`hostname -I|awk -F " " \'{print$2}\'`',
        'echo "${LOCAL_HOST1}"',
        'echo "${LOCAL_HOST2}"',
    ])

    # Prefill loop
    lines.append("# prefill")
    lines.append('for i in "${!P_IP[@]}";')
    lines.append("do")
    lines.append('    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];')
    lines.append("    then")
    lines.append('        echo "${P_IP[$i]}"')
    for env_line in format_env_exports(prefill_envs_filtered).split("\n"):
        if env_line.strip():
            lines.append(f"        {env_line}")
    lines.append("")

    # Build prefill command line
    prefill_flags = [f"--host ${{P_IP[$i]}}", "--port 8000"]
    if prefill_nnodes > 1:
        if pd_prefill > 1:
            nn = prefill_nnodes
            prefill_flags.append(f"--dist-init-addr ${{P_IP[$(( $i / {nn} * {nn} ))]}}:5000")
        else:
            prefill_flags.append("--dist-init-addr ${P_IP[0]}:5000")
    if pd_prefill > 1:
        if prefill_nnodes > 1:
            nn = prefill_nnodes
            prefill_flags.append(f"--disaggregation-bootstrap-port $((8998 + $i / {nn}))")
        else:
            prefill_flags.append("--disaggregation-bootstrap-port $((8998 + $i))")
    else:
        prefill_flags.append("--disaggregation-bootstrap-port 8998")
    if had_nnodes:
        if prefill_nnodes > 1:
            if pd_prefill > 1:
                nn = prefill_nnodes
                prefill_flags.append(f"--node-rank $(( $i % {nn} ))")
            else:
                prefill_flags.append("--node-rank $i")
        else:
            prefill_flags.append("--node-rank 0")

    all_flags = " \\\n        ".join(prefill_flags)
    if prefill_args_str:
        all_flags += f" \\\n        {prefill_args_str}"
    lines.append(f"        python3 -m sglang.launch_server \\")
    lines.append(f"        --model-path ${{MODEL_PATH}} \\")
    lines.append(f"        --disaggregation-mode prefill \\")
    lines.append(f"        {all_flags}")
    lines.append("        NODE_RANK=$i")
    lines.append("        break")
    lines.append("    fi")
    lines.append("done")
    lines.append("")

    # Decode loop
    lines.append("# decode")
    lines.append('for i in "${!D_IP[@]}";')
    lines.append("do")
    lines.append('    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];')
    lines.append("    then")
    lines.append('        echo "${D_IP[$i]}"')
    for env_line in format_env_exports(decode_envs_filtered).split("\n"):
        if env_line.strip():
            lines.append(f"        {env_line}")
    lines.append("")

    # Build decode command line
    decode_flags = [f"--host ${{D_IP[$i]}}", "--port 8001"]
    if decode_nnodes > 1:
        decode_flags.append("--dist-init-addr ${D_IP[0]}:5000")
    if decode_nnodes > 1:
        decode_flags.append("--node-rank $i")

    all_dflags = " \\\n        ".join(decode_flags)
    if decode_args_str:
        all_dflags += f" \\\n        {decode_args_str}"
    lines.append(f"        python3 -m sglang.launch_server \\")
    lines.append(f"        --model-path ${{MODEL_PATH}} \\")
    lines.append(f"        --disaggregation-mode decode \\")
    lines.append(f"        {all_dflags}")
    lines.append("        NODE_RANK=$i")
    lines.append("        break")
    lines.append("    fi")
    lines.append("done")

    deploy_cmd = "\n".join(comment_lines + [""] + lines)

    # Router command (always output for PD-separate)
    if pd_prefill > 1:
        ip_list = ", ".join(f"<your prefill ip{i+1}>" for i in range(pd_prefill))
        router_comment_items = [
            "# Before running, replace the following placeholders:",
            f"#   {ip_list}: prefill node IP addresses",
        ]
    else:
        router_comment_items = [
            "# Before running, replace the following placeholders:",
            "#   <your prefill ip>: prefill node IP address",
        ]
    if d_ip_count > 1:
        router_comment_items.append("#   <your decode ip1>: first decode node IP address (decode may have distributed nodes)")
    else:
        router_comment_items.append("#   <your decode ip>: decode node IP address")
    router_lines = [
        "# ============================================================",
    ] + router_comment_items + [
        "# ============================================================",
        "",
    ]
    for k, v in sorted(router_envs.items()):
        if k in EXCLUDED_ENV_VARS:
            continue
        if _env_value_needs_quoting(v):
            v = '"' + v.replace('"', '\\"') + '"'
        router_lines.append(f"export {k}={v}")
    router_args_str = " "
    if router_args:
        router_args_str += " ".join([a for a in router_args if a])
    router_lines.append("python -m sglang_router.launch_router \\")
    router_lines.append("    --pd-disaggregation \\")
    has_router_policy = "--policy" in router_args
    if not has_router_policy:
        router_lines.append("    --policy cache_aware \\")
    for g in range(pd_prefill):
        ip_label = f"<your prefill ip{g+1}>" if pd_prefill > 1 else "<your prefill ip>"
        router_lines.append(f"    --prefill http://{ip_label}:8000 {8998 + g} \\")
    decode_label = "<your decode ip1>" if d_ip_count > 1 else "<your decode ip>"
    router_lines.append(f"    --decode http://{decode_label}:8001 \\")
    router_lines.append("    --host 127.0.0.1 \\")
    router_lines.append("    --port 6688 \\")
    router_lines.append(f"    {router_args_str.strip()}")
    router_cmd = "\n".join(router_lines)

    return deploy_cmd, router_cmd


def format_single_node_command(config):
    envs = config.get("envs", {})
    other_args = config.get("other_args", [])

    envs_filtered = {}
    for k, v in sorted(envs.items()):
        if k in ("HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
            envs_filtered[k] = "<network-interface>"
        else:
            envs_filtered[k] = v

    has_pythonpath = "PYTHONPATH" in envs_filtered
    if has_pythonpath:
        del envs_filtered["PYTHONPATH"]

    env_block = format_env_exports(envs_filtered)

    nnodes = 1
    for i, arg in enumerate(other_args):
        if arg == "--nnodes" and i + 1 < len(other_args):
            try:
                nnodes = int(other_args[i + 1])
            except (ValueError, TypeError):
                pass

    is_multi_node = nnodes > 1

    if is_multi_node:
        new_args = []
        skip_next = False
        for a in other_args:
            if skip_next:
                skip_next = False
                continue
            if a == "--nnodes":
                skip_next = True
                continue
            new_args.append(a)
        args_str = format_args_for_bash(new_args, indent="        ")
    else:
        args_str = format_args_for_bash(other_args, indent="    ")

    has_draft = "--speculative-draft-model-path" in other_args

    comment_items = [
        "#   MODEL_PATH: path to the model weights directory",
    ]
    if has_draft:
        comment_items.append("#   DRAFT_MODEL_PATH: path to the draft model weights directory")
    if is_multi_node:
        comment_items.append("#   NODE_IPS: IP addresses of each node in the cluster")
    comment_items += [
        "#   HCCL_SOCKET_IFNAME: network interface name for HCCL",
        "#   GLOO_SOCKET_IFNAME: network interface name for Gloo",
    ]
    header = "\n".join([
        "# ============================================================",
        "# Before running, update the following variables:",
    ] + comment_items + [
        "# ============================================================",
    ])

    if has_draft:
        draft_var = "DRAFT_MODEL_PATH=/path/to/draft-model-weights\n"
        if has_pythonpath:
            draft_var += "export PYTHONPATH=${DRAFT_MODEL_PATH}:$PYTHONPATH\n"
        draft_var += "\n"
    else:
        draft_var = ""

    if is_multi_node:
        ips = " ".join(f"'<your node{i+1} ip>'" for i in range(nnodes))
        cmd = f"""{header}

MODEL_PATH=/path/to/model-weights
{draft_var}NODE_IPS=({ips})

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

{env_block}

LOCAL_HOST1=`hostname -I|awk -F " " '{{print$1}}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{{print$2}}'`
echo "${{LOCAL_HOST1}}"
echo "${{LOCAL_HOST2}}"

for i in "${{!NODE_IPS[@]}}";
do
    if [[ "$LOCAL_HOST1" == "${{NODE_IPS[$i]}}" || "$LOCAL_HOST2" == "${{NODE_IPS[$i]}}" ]];
    then
        echo "${{NODE_IPS[$i]}}"
        python3 -m sglang.launch_server \\
        --model-path $MODEL_PATH \\
        --host ${{NODE_IPS[$i]}} --port 6688 \\
        --nnodes {nnodes} \\
        --dist-init-addr ${{NODE_IPS[0]}}:5000 \\
        --node-rank $i \\
        {args_str}
        break
    fi
done
"""
    else:
        cmd = f"""{header}

MODEL_PATH=/path/to/model-weights
{draft_var}echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

{env_block}

python3 -m sglang.launch_server \\
    --model-path $MODEL_PATH \\
    --host 127.0.0.1 --port 6688 \\
    {args_str}
"""
    return cmd, None


def format_benchmark_command(config):
    bm = config["benchmark"]
    if not bm:
        return "python -m sglang.bench_serving --dataset-name random --backend sglang"

    dataset_name = bm.get("dataset_name", "random")
    backend = bm.get("backend", "sglang")

    parts = [
        "python -m sglang.bench_serving",
        f"--dataset-name {dataset_name}",
        f"--backend {backend}",
        "--host 127.0.0.1",
        "--port 6688",
    ]

    # GSP dataset needs special handling.
    if dataset_name == "generated-shared-prefix":
        repeat_rate = float(bm.get("repeat_rate", 0.9))
        input_len = int(bm.get("input_len", 0))
        output_len = int(bm.get("output_len", 0))
        num_prompts = int(bm.get("num_prompts", 0))
        gsp_system_prompt_len = round(repeat_rate * input_len)
        gsp_question_len = round((1 - repeat_rate) * input_len)
        parts.append("--gsp-num-groups 1")
        if num_prompts:
            parts.append(f"--gsp-prompts-per-group {num_prompts}")
        if gsp_system_prompt_len:
            parts.append(f"--gsp-system-prompt-len {gsp_system_prompt_len}")
        if gsp_question_len:
            parts.append(f"--gsp-question-len {gsp_question_len}")
        if output_len:
            parts.append(f"--gsp-output-len {output_len}")
        if "max_concurrency" in bm:
            parts.append(f"--max-concurrency {safe_val(bm['max_concurrency'])}")
        if num_prompts:
            parts.append(f"--num-prompts {num_prompts}")
        if "request_rate" in bm:
            parts.append(f"--request-rate {safe_val(bm['request_rate'])}")
        if "seed" in bm:
            parts.append(f"--seed {safe_val(bm['seed'])}")
        return " \\\n    ".join(parts)

    # Dynamic benchmark flag mapping.
    #  - Keys with a string value → explicit ``--flag``.
    #  - Keys with a ``_flag`` entry → override the auto-generated flag name.
    #  - Any other key in ``bm`` (not listed below) is auto-mapped:
    #    ``field_name`` → ``--field-name``.
    _BM_FLAG_MAP = {
        "random_range_ratio": "--random-range-ratio",
        "input_len":          "--random-input-len",
        "output_len":         "--random-output-len",
        "warmup_requests":    "--warmup-requests",
        "request_rate":       "--request-rate",
        "num_prompts":        "--num-prompts",
        "max_concurrency":    "--max-concurrency",
        "seed":               "--seed",
    }

    # Image-specific flags.
    if dataset_name == "image":
        _BM_FLAG_MAP["image_count"] = "--image-count"
        _BM_FLAG_MAP["image_resolution"] = "--image-resolution"

    # Fields that should never appear in the benchmark command.
    _skip = {
        "repeat_rate", "tpot", "ttft", "mean_e2e_latency",
        "output_token_throughput", "dataset_name", "backend", "dataset_type",
        "model", "model_path", "other_args", "envs", "model_config",
        "other_envs", "model_type",
    }

    for key, val in bm.items():
        if key in _skip:
            continue
        if val is None or val == "":
            continue
        if key in _BM_FLAG_MAP:
            flag = _BM_FLAG_MAP[key]
        else:
            # Auto-derive flag name: foo_bar → --foo-bar
            flag = "--" + key.replace("_", "-")
        parts.append(f"{flag} {safe_val(val)}")

    return " \\\n    ".join(parts)


def _get_sort_metric(bm):
    """Return the primary latency metric value for category ordering."""
    if "tpot" in bm:
        return bm["tpot"]
    if "ttft" in bm:
        return bm["ttft"]
    return 999


def _get_metric_display(bm):
    """Return (tpot_str, ttft_str) for display in tables and headings."""
    tpot_val = bm.get("tpot")
    ttft_val = bm.get("ttft")
    tpot_str = f"{tpot_val}ms" if tpot_val is not None else ""
    if ttft_val is not None:
        # ttft values are in ms; display as "s" when >= 1000
        ttft_str = f"{ttft_val / 1000:.3g}s" if ttft_val >= 1000 else f"{ttft_val}ms"
    else:
        ttft_str = ""
    return tpot_str, ttft_str


def _has_ttft(configs):
    """Return True if any config in the list has a ttft field (even alongside tpot)."""
    return any("ttft" in c.get("benchmark", {}) for c in configs)


def generate_anchor(heading):
    """Generate anchor from heading text, matching Docusaurus auto-generated slug."""
    slug = heading.lower()
    slug = slug.replace(".", "-")
    slug = re.sub(r'[^a-z0-9_\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'-{2,}', '-', slug)
    slug = slug.strip('-')
    return slug

def generate_heading_label(config, model_name, model_dir):
    """Generate heading text whose auto-generated ID matches the anchor link."""
    filename = config["filename"].replace("test_npu_", "").replace(".py", "")
    filename = re.sub(r'_(gpqa|mmlu|aime\d*)$', '', filename.lower())
    # Strip the model identifier prefix from filename by finding the common prefix
    # over underscore-separated segments.
    model_parts = model_dir.replace("-", "_").split("_")
    file_parts = filename.split("_")
    common_len = 0
    for mp, fp in zip(model_parts, file_parts):
        if mp == fp:
            common_len += 1
        else:
            break
    if common_len > 0:
        config_part = "_".join(file_parts[common_len:])
    else:
        config_part = filename
    tpot = config.get("benchmark", {}).get("tpot")
    if tpot is not None:
        tpot_str = str(tpot)
        config_part = re.sub(r'_\d+(?:\.\d+)?ms$', f'_{tpot_str}ms', config_part)
    config_part = config_part.upper().replace("_", " ")
    config_part = re.sub(r'(\d+)MS', r'\1ms', config_part)
    config_part = re.sub(r'(\d+)S\b', r'\1s', config_part)
    heading = f"{model_name} {config_part}"
    return heading


def build_model_document(model_dir, configs):
    model_name = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)

    lines = []
    lines.append("---")
    lines.append(f'title: "{model_name}"')
    lines.append("metatags:")
    lines.append(f'  description: "Best Practice for {model_name} on Ascend NPU"')
    lines.append("---")
    lines.append("")

    # Link to the corresponding model tutorial (user journey) when it exists.
    tutorial_slug = get_tutorial_slug(model_dir)
    tutorial_path = os.path.normpath(
        os.path.join(OUTPUT_DIR, "..", "model-tutorials", f"{tutorial_slug}.mdx")
    )
    has_tutorial = os.path.isfile(tutorial_path)
    tutorial_url = f"/docs/hardware-platforms/ascend-npus/model-tutorials/{tutorial_slug}"

    if has_tutorial:
        lines.append("<Note>")
        lines.append(
            f"This page focuses on optimal configuration and benchmark results for {model_name} on the Ascend NPU. "
            f"For environment setup, model weight download, feature configuration, and deployment instructions, etc., "
            f"see the [{model_name} Model Tutorial]({tutorial_url})."
        )
        lines.append("")
        lines.append(
            "On A3 each card has 2 dies, so `--tp-size` is twice the card count; "
            "see [Ascend NPU Reference](/docs/hardware-platforms/ascend-npus/ascend_npu_reference#hardware) for details."
        )
        lines.append("</Note>")
    else:
        lines.append("<Note>")
        lines.append(
            f"This page focuses on optimal configuration and benchmark results for {model_name} on the Ascend NPU."
        )
        lines.append("")
        lines.append(
            "On A3 each card has 2 dies, so `--tp-size` is twice the card count; "
            "see [Ascend NPU Reference](/docs/hardware-platforms/ascend-npus/ascend_npu_reference#hardware) for details."
        )
        lines.append("</Note>")
    lines.append("")

    # Separate configs by category.
    # TTFT configs always go to High Throughput (they measure first-token latency,
    # not per-token latency).
    low_latency = [c for c in configs if "tpot" in c.get("benchmark", {})
                   and c["benchmark"]["tpot"] < 30]
    high_throughput = [c for c in configs if c not in low_latency]

    has_ttft = _has_ttft(low_latency + high_throughput)

    def write_table(configs_list, title):
        if not configs_list:
            return

        lines.append(f"### {title}")
        lines.append("")
        headers = ["Model", "Hardware", "Cards", "Deploy Mode", "Dataset", "TPOT"]
        if has_ttft:
            headers.append("TTFT")
        headers += ["Quantization", "Configuration"]
        # Header row
        lines.append("| " + " | ".join(headers) + " |")
        # Separator row
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for c in configs_list:
            heading_label = generate_heading_label(c, model_name, model_dir)
            anchor = generate_anchor(heading_label)
            bm = c.get("benchmark", {})
            dataset = parse_dataset_from_filename(c.get("filename", ""))
            tpot_str, ttft_str = _get_metric_display(bm)
            tpot_str = tpot_str or "-"
            ttft_str = ttft_str or "-"

            cells = [
                model_name,
                c.get("hardware", "Atlas 800I A3"),
                str(c.get("cards", "")),
                c.get("deploy_mode", "PD Mixed"),
                dataset,
                tpot_str,
            ]
            if has_ttft:
                cells.append(ttft_str)
            cells += [
                c.get("quantization", "W8A8 INT8"),
                f"[Optimal Configuration](#{anchor})",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    write_table(low_latency, "Low Latency")
    write_table(high_throughput, "High Throughput")

    # Optimal Configuration sections
    lines.append("## Optimal Configuration")
    lines.append("")

    seen_anchor_types = set()

    # Determine PD disaggregation anchor placement:
    #  - prefer multi-node PD -> anchor before first multi-node PD config.
    #  - fall back to single-node PD -> anchor before first single-node PD config.
    has_multi_pd = any(c.get("is_pd_separate") and c.get("is_multi_node") for c in configs)
    has_single_pd = any(c.get("is_pd_separate") and not c.get("is_multi_node") for c in configs)
    pd_anchor_emitted = False

    for c in configs:
        bm = c.get("benchmark", {})
        dataset = parse_dataset_from_filename(c.get("filename", ""))
        heading_label = generate_heading_label(c, model_name, model_dir)
        tpot_str, ttft_str = _get_metric_display(bm)

        is_sep = c.get("is_pd_separate", False)
        is_multi = c.get("is_multi_node", False)
        if is_sep:
            if has_multi_pd:
                # Anchor before first multi-node PD disaggregation.
                anchor_type = "pd-disaggregation" if (is_multi and not pd_anchor_emitted) else None
            else:
                # No multi-node PD -> anchor before first single-node PD disaggregation.
                anchor_type = "pd-disaggregation" if (not is_multi and not pd_anchor_emitted) else None
            if anchor_type:
                pd_anchor_emitted = True
        elif is_multi:
            anchor_type = "multi-node-pd-mixed"
        else:
            anchor_type = "single-node-pd-mixed"

        if anchor_type and anchor_type not in seen_anchor_types:
            lines.append(f'<a id="{anchor_type}" title="Referenced by external docs. Verify before removing."></a>')
            lines.append("")
            seen_anchor_types.add(anchor_type)

        lines.append(f"### {heading_label}")
        lines.append("")
        lines.append(f"**Model**: {model_name}")
        lines.append("")
        lines.append(f"**Hardware**: {c.get('hardware', 'Atlas 800I A3')}")
        lines.append("")
        lines.append(f"**Cards**: {c.get('cards', '')}")
        lines.append("")
        lines.append(f"**Deploy Mode**: {c.get('deploy_mode', 'PD Mixed')}")
        lines.append("")
        lines.append(f"**Quantization**: {c.get('quantization', 'W8A8 INT8')}")
        lines.append("")
        lines.append(f"**Dataset**: {dataset}")
        if re.search(r'\d+x\d+', dataset):
            lines.append("")
            lines.append("*Format: resolution (input tokens) + output tokens*")
        lines.append("")
        if tpot_str:
            lines.append(f"**TPOT**: {tpot_str}")
            if ttft_str:
                lines.append("")
        if ttft_str:
            lines.append(f"**TTFT**: {ttft_str}")
        lines.append("")

        lines.append("#### Model Deployment")
        lines.append("")

        if c.get("is_pd_separate"):
            deploy_cmd, router_cmd = format_pd_separate_command(c)
        else:
            deploy_cmd, router_cmd = format_single_node_command(c)

        lines.append("```bash Command")
        lines.append(deploy_cmd.rstrip())
        lines.append("```")
        lines.append("")

        if router_cmd:
            lines.append("```bash Command")
            lines.append(router_cmd.rstrip())
            lines.append("```")
            lines.append("")

        lines.append("#### Benchmark")
        lines.append("")
        dataset_name = c.get("benchmark", {}).get("dataset_name", "random")
        if dataset_name == "generated-shared-prefix":
            repeat_rate = float(c.get("benchmark", {}).get("repeat_rate", 0.9))
            input_len = int(c.get("benchmark", {}).get("input_len", 0))
            pct = int(repeat_rate * 100)
            gsp_system_prompt_len = round(repeat_rate * input_len)
            gsp_question_len = round((1 - repeat_rate) * input_len)
            desc = f"We tested it based on the `{dataset_name}` dataset with {pct}% cache hit (`repeat_rate = {repeat_rate}`):\n"
            desc += f"`--gsp-system-prompt-len {gsp_system_prompt_len}` = `round({input_len} * {repeat_rate})` is the shared prefix portion.\n"
            desc += f"`--gsp-question-len {gsp_question_len}` = `round({input_len} * (1 - {repeat_rate}))` is the unique per-request suffix.\n"
            desc += f"`--gsp-num-groups 1` keeps all requests in one prefix group for maximum cache reuse."
        elif dataset_name == "image":
            resolution = c.get("benchmark", {}).get("image_resolution", "")
            if resolution:
                desc = f"We tested it based on the `IMAGE` dataset with {resolution} resolution."
            else:
                desc = f"We tested it based on the `IMAGE` dataset."
        else:
            desc = f"We tested it based on the `{dataset_name.upper()}` dataset."
        lines.append(desc)
        lines.append("")
        lines.append("```bash Command")
        lines.append(format_benchmark_command(c))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_dir in sorted(os.listdir(PERFORMANCE_DIR)):
        model_path = os.path.join(PERFORMANCE_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue

        configs = []
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(model_path, fname)
            try:
                config = extract_config_from_file(fpath)
                configs.append(config)
            except Exception as e:
                print(f"Warning: Error parsing {fpath}: {e}", file=sys.stderr)

        if not configs:
            print(f"No configs found for {model_dir}")
            continue

        # Only include configs that have benchmark parameters (skip accuracy tests)
        valid_configs = [c for c in configs if c.get("benchmark", {}).get("tpot") or c.get("benchmark", {}).get("ttft")]
        if not valid_configs:
            print(f"No valid benchmark configs for {model_dir}")
            continue

        content = build_model_document(model_dir, valid_configs)
        output_file = os.path.join(OUTPUT_DIR, f"{model_dir}.mdx")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Generated {output_file} with {len(valid_configs)} configs")


if __name__ == "__main__":
    main()

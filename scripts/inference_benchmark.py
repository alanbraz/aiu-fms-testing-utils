import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict

from aiu_fms_testing_utils.utils import aiu_setup
from aiu_fms_testing_utils.utils.aiu_setup import dprint, rank, local_rank, world_size
import numpy as np
import torch._inductor.config
from fms.models import get_model, register_model
from fms.models.llama import LLaMAConfig, _llama_factory_factory
from fms.utils import fusion, generation, tokenizers
from fms.utils.generation import generate, pad_input_ids
from torch import distributed as dist

# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
)
parser.add_argument(
    "--device_type",
    type=str,
    choices=["cuda", "cpu", "aiu", "aiu-senulator"],
    default="cuda",
    help="The device to run the model on"
)
parser.add_argument(
    "--architecture",
    type=str,
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default=None,
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--quantization",
    type=str,
    choices=["gptq"],
    default=None,
    help="Type of quantization of the model checkpoint",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--unfuse_weights",
    action="store_true",
    help="If set to True, this will unfuse any fused weight modules that support the unfuse_weights method",
)
parser.add_argument(
    "--default_dtype",
    type=str,
    default=None,
    choices=["bf16", "fp16", "fp32"],
    help="If set to one of the choices, overrides the model checkpoint weight format by setting the default pytorch format",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation (only valid for inductor backend)",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--compile_backend",
    type=str,
    help="Backend for compilation (only when not running on AIU)",
    default="inductor",
    choices=["inductor", "eager", "aot_eager"],
)
parser.add_argument(
    "--compile_dynamic",
    action="store_true",
    help="Use dynamic shapes with torch.compile",
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="size of input batch",
)
parser.add_argument(
    "--max_prompt_length",
    type=int,
    default=None,
    help="cap the number of tokens per prompt to a maximum length prior to padding. If None, there will be no cap.",
)
parser.add_argument(
    "--min_pad_length",
    type=int,
    help="Pad inputs to a minimum specified length. If any prompt is larger than the specified length, padding will be determined by the largest prompt",
    default=0,
)
parser.add_argument(
    "--fixed_prompt_length",
    type=int,
    help="If defined, overrides both min_pad_length and max_prompt_length. Pads input to fixed_prompt_length, fails if any input needs truncation.",
    default=0,
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="max number of generated tokens",
    default=100,
)
parser.add_argument(
    "--no_early_termination",
    action="store_true",
    help="disable early termination on generation",
)
parser.add_argument(
    "--prompt_type",
    type=str,
    choices=["chat", "code"],
    default="chat",
    help="type of prompts to be used, either chat or code",
)
parser.add_argument(
    "--prompt_path",
    type=str,
    default="",
    help="if set, load the prompts from file(s) instead of the local examples. Supports glob-style patterns",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="",
    help="path of folder to save outputs to, if empty don't save",
)

parser.add_argument(
    "--json_output_file",
    type=str,
    default="",
    help="path of file to save json outputs to, if empty don't save",
)
parser.add_argument(
    "--timing",
    type=str,
    choices=["e2e", "per-token"],
    default="",
    help="if set, how to time the generation of tokens, e2e or per-token",
)
parser.add_argument(
    "--iters",
    type=int,
    default=1,
    help="Number of iterations of inference to perform. Used for variance performance capture.",
)
parser.add_argument(
    '-v', '--verbose',
    action='count',
    default=0,
    help="Set verbosity level (pass flag as `-v`, `-vv`, `-vvv`)"
)
args = parser.parse_args()

if args.quantization == "gptq":
    GPTQ_ENABLED = True
    try:
        if "aiu" in args.device_type:
            from fms_mo.aiu_addons.gptq import gptq_aiu_adapter, gptq_aiu_linear
            print("Loaded `aiu_addons` functionalities")
        elif args.device_type != "cpu":
            raise ValueError(f"Device {args.device_type} unsupported for GPTQ run")
    except ImportError as e:
        print(f"Failed to import addon packages: {e}")
        GPTQ_ENABLED = False

    if not GPTQ_ENABLED:
        raise Exception("GPTQ not enabled")

# this is a test model config
config = LLaMAConfig(
    emb_dim=1024,
    nheads=8,
    nlayers=10,
    src_vocab_size=128256,
)
register_model("llama", "194m", _llama_factory_factory(config))

default_dtype = None
dtypes_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
if args.default_dtype is not None:
    default_dtype = dtypes_map[args.default_dtype]

if default_dtype is not None:
    torch.set_default_dtype(default_dtype)

dprint(f"{args}")

benchmark_json: Dict[str, Any] = {}

benchmark_json["cluster"] = "ais5"
benchmark_json["device_type"] = args.device_type
benchmark_json["num_aius"] = os.getenv("NUM_AIUS", None)
try:
    import subprocess
    bash_command = "/opt/sentient/bin/aiu-query-devices --skip-topo | tail -n +2"
    result = subprocess.check_output(bash_command, shell=True, text=True)
    benchmark_json["devices-info"] = result.split("\n")
except:
    benchmark_json["devices-info"] = []
benchmark_json["precision"] = args.default_dtype
benchmark_json["batch_size"] = args.batch_size
benchmark_json["max_prompt_length"] = args.max_prompt_length
benchmark_json["max_new_tokens"] = args.max_new_tokens

output_json = vars(args)
output_json["e2e_s"] = []
output_json["ttft_s"] = []
output_json["itls_s"] = []
output_json["tpot_ms"] = []
output_json["itl_ms"] = []
output_json["prompts_raw"] = []
output_json["prompts_raw_len"] = []
output_json["prompts"] = []
output_json["prompts_len"] = []
output_json["responses"] = []
output_json["responses_len"] = []
output_json["warmup_e2e_s"] = []
output_json["warmup_ttft_s"] = []
output_json["warmup_itls_s"] = []
output_json["warmup_tpot_ms"] = []
output_json["warmup_itl_ms"] = []
output_json["warmup_responses_len"] = []

is_aiu_backend = "aiu" in args.device_type

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    aiu_setup.aiu_dist_setup(dist.get_rank(), dist.get_world_size())

if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
elif is_aiu_backend:
    from torch_sendnn import torch_sendnn

    if not args.distributed:
        aiu_setup.aiu_setup(rank, world_size)

    _target_cache_size = max(
        int(args.max_new_tokens * 2),
        int(args.min_pad_length * 2.5),
        int(args.fixed_prompt_length * 2.5),
    )
    _prompt_size = max(int(args.min_pad_length), int(args.fixed_prompt_length))
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        if _target_cache_size > torch._dynamo.config.accumulated_cache_size_limit:
            _prev = torch._dynamo.config.accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = _target_cache_size
            dprint(
                f"NOTICE: Adjusting torch._dynamo.config.accumulated_cache_size_limit from {_prev} to {torch._dynamo.config.accumulated_cache_size_limit} to accomodate prompt size of {_prompt_size} and decode tokens of {args.max_new_tokens}"
            )

    if _target_cache_size > torch._dynamo.config.cache_size_limit:
        _prev = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = _target_cache_size
        dprint(
            f"NOTICE: Adjusting torch._dynamo.config.cache_size_limit from {_prev} to {torch._dynamo.config.cache_size_limit} to accomodate prompt size of {_prompt_size} and decode tokens of {args.max_new_tokens}"
        )

    if not args.compile_dynamic:
        torch._dynamo.config.assume_static_by_default = True
        torch._dynamo.config.dynamic_shapes = False
        torch._dynamo.config.automatic_dynamic_shapes = False

    # This should be set outside!!!
    os.environ.setdefault("SENCORES", "32")
    os.environ.setdefault("SENCORELETS", "2")
    os.environ.setdefault("DATA_PREC", "fp16")
    os.environ.setdefault("FLEX_OVERWRITE_NMB_FRAME", "1")
    os.environ.setdefault("DTCOMPILER_KEEP_EXPORT", "true")

    os.environ.setdefault("COMPILATION_MODE", "offline_decoder")

    if args.device_type == "aiu-senulator":
        os.environ["FLEX_COMPUTE"] = "SENULATOR"
        os.environ["FLEX_DEVICE"] = "MOCK"
    else:
        if "AIU_WORLD_RANK_0" not in os.environ:
            print("must set AIU_WORLD_RANK_0")
            exit()
        os.environ["FLEX_COMPUTE"] = "SENTIENT"
        os.environ["FLEX_DEVICE"] = "VFIO"

    device = torch.device("cpu")
else:
    device = torch.device(args.device_type)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

dprint("loading model")
loading_model_time = time.time()
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

fused_weights = not args.unfuse_weights
if args.quantization == "gptq":
    if "aiu" in args.device_type:
        linear_type = "gptq_aiu"
    elif args.device_type == "cpu":
        linear_type = "gptq_cpu"
    elif args.device_type == "cuda":
        linear_type = "gptq"  # GPTQ support on GPU is FMS-native
    else:
        raise ValueError(f"Unsupported device {args.device} for GPTQ")

    qconfig_path = args.model_path + "/quantize_config.json"
    if os.path.exists(qconfig_path):
        with open(qconfig_path, 'r') as f:
            dprint(f"loading quantization config from {qconfig_path}")
            qconfig = json.load(f)
            group_size = qconfig["group_size"]
            desc_act = qconfig["desc_act"]
            if desc_act:
                raise NotImplementedError(
                    "Activation reordering not supported at this time."
                )
    else:
        dprint(
            "[WARNING] Could not locate quantization config file. "
            "Default configuration will be used."
        )
        group_size = 128
        desc_act = False

    linear_config = {
        "linear_type": linear_type,
        "group_size": group_size,
        "desc_act": desc_act,
    }
    # [ATTENTION] for GPTQ on AIU, we must always instantiate an unfused
    # model, the adapter will take care of converting key/values from
    # ckpt into the appropriate form for the model
    if fused_weights:
        raise ValueError("GPTQ checkpoints on AIU must always run with --unfuse_weights")
    default_dtype = None  # GPTQ dtype always comes from ckpt, can't be enforced
else:
    linear_config = {"linear_type": "torch_linear"}

dprint("="*60)
dprint(f"model_path={args.model_path}")
dprint(f"{linear_config=}")
dprint(f"{fused_weights=}")
dprint(f"data_type={default_dtype}")
dprint("="*60 + "\n")

benchmark_start_time = time.perf_counter()

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type="cpu" if is_aiu_backend else args.device_type,
    data_type=default_dtype,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
    linear_config=linear_config,
    fused_weights=fused_weights,
)

if args.quantization == "gptq":
    if rank == 0 and args.verbose > 0:
        dprint("PARAMS:\n" + "\n".join(f"{k:60} {str(v.dtype):15} {str(v.device):10} {list(v.size())}" for k,v in model.named_parameters()))
        dprint("BUFFERS:\n" + "\n".join(f"{k:60} {str(v.dtype):15} {str(v.device):10} {list(v.size())}" for k,v in model.named_buffers()))
        dprint("="*60 + "\n")
    if args.architecture == "llama":
        dprint("[NOTE] It's OK for unused keys to contain bias and rotary embeddings, in GPTQ LLaMA models")
    dprint(model)
    dprint("="*60 + "\n")

tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
loading_model_time = time.time() - loading_model_time
dprint(f"loading complete, took {loading_model_time:.3f}s")

benchmark_json["model_load_s"] = loading_model_time

if args.compile:
    dprint("compiling model")
    if is_aiu_backend:
        model.compile(backend="sendnn_decoder")
    else:
        # compiling can make first inference pass slow
        model.compile(mode=args.compile_mode, backend=args.compile_backend)

add_special_tokens = tokenizer.bos_token_id != tokenizer.eos_token_id

def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if add_special_tokens:
        ids = [tokenizer.bos_token_id] + ids
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids


def truncate_prompts_to_max_length(prompts, max_len, max_allowed_length):
    # we may want the prompt length to be fixed to some max length
    # this will ensure that prior to padding the input ids
    if max_allowed_length is not None and max_len > max_allowed_length:
        dprint(f"max prompt length is {max_len}, truncating to {max_allowed_length}")
        prompts = [prompt[:max_allowed_length] for prompt in prompts]
    return prompts


if args.prompt_path != "":
    # Before creating the Path object, check if prompt_path has a glob pattern
    if isinstance(args.prompt_path, str):
        prompt_path, sep, glob_pattern = args.prompt_path.partition("*")
    else:
        sep = ""
        glob_pattern = ""
    glob_pattern = sep + glob_pattern

    prompt_path = Path(os.path.expanduser(prompt_path))
    prompt_file_paths = []

    if prompt_path.is_dir():
        if glob_pattern != "":
            glob_pattern_list = [glob_pattern]
        else:
            glob_pattern_list = ["*.txt"]
        for glob_pattern_possibility in glob_pattern_list:
            file_list = list(prompt_path.glob(glob_pattern_possibility))
            if len(file_list) > 0:
                prompt_file_paths = sorted(file_list)
                break

    if prompt_path.is_file():
        prompt_file_paths = [prompt_path]

    # Check if we found some files
    assert len(prompt_file_paths) > 0, f"Can't find any prompt files at {prompt_path}"

    # Check if we have enough files
    assert (
        len(prompt_file_paths) >= args.batch_size
    ), f"Not enough prompt files at {prompt_path} for a batch size of {args.batch_size}"

    prompts = []
    for i, prompt_file_path in enumerate(prompt_file_paths):
        if i == args.batch_size * args.iters:
            break
        prompts.append(ids_for_prompt(prompt_file_path.read_text(encoding="utf-8")))

else:
    if args.prompt_type == "chat":
        template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

        prompt1 = template.format(
            "Provide a list of instructions for preparing chicken soup."
        )
        prompt2 = template.format("Explain some popular greetings in Spanish.")
        prompt3 = template.format("Explain to me why ignorance is bliss.")
        prompt4 = template.format(
            "I have just come into a very large sum of money. Provide me a list of things that I can do with my new found wealth."
        )
    elif args.prompt_type == "code":
        template = "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{}\n[/INST]"
        prompt1 = template.format("Write a bubble sort function in python.")
        prompt2 = template.format(
            "Using the Java streams API, write a simple function which will get the cumulative sum of a list of integers."
        )
        prompt3 = template.format(
            "In bash, how do I list all directories and sub-directories which contain a .py file."
        )
        prompt4 = template.format(
            "Write a simple decorator in python which will modify all string inputs to ints if possible."
        )
    else:
        dprint("prompt_type must be one of chat or code")
        exit()

    prompt1 = ids_for_prompt(prompt1)
    prompt2 = ids_for_prompt(prompt2)
    prompt3 = ids_for_prompt(prompt3)
    prompt4 = ids_for_prompt(prompt4)
    prompts = [prompt1, prompt2, prompt3, prompt4]
    prompts = prompts * ((args.batch_size // 4) + 1)
    prompts = prompts[: args.batch_size]

for prompt in prompts:
    output_json["prompts_raw"].append(tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(prompt)
    ))
    output_json["prompts_raw_len"].append(len(prompt))
                                              
if args.fixed_prompt_length != 0:
    padding_length = args.fixed_prompt_length
    max_allowed_length = args.fixed_prompt_length
else:
    padding_length = args.min_pad_length
    max_allowed_length = args.max_prompt_length

has_padding = args.batch_size > 1 or padding_length != 0
max_len = max([len(prompt) for prompt in prompts])

if args.fixed_prompt_length != 0 and args.fixed_prompt_length < max_len:
    dprint(
        f"One or more prompts require truncation. Truncation has been disabled as fixed_prompt_length has been set."
    )
    exit(1)
prompts = truncate_prompts_to_max_length(prompts, max_len, max_allowed_length)

for prompt in prompts:
    output_json["prompts"].append(tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(prompt)
    ))
    output_json["prompts_len"].append(len(prompt))

# if has_padding:
#     ids, extra_generation_kwargs = pad_input_ids([prompts[0]], min_pad_length=padding_length)
# else:
#     ids = prompts
#     extra_generation_kwargs = None

def print_result(result, result_idx: int):
    if local_rank != 0:
        return
    if has_padding:
        result = generation.trim_prefix(result)

    result = generation.trim_prefix(result, tokenizer.bos_token_id)

    # stop at EOS token if present and remove padding
    if not args.no_early_termination:
        result = generation.truncate_after_eos(result, tokenizer.eos_token_id)

    output_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(result)
    )

    if args.output_path != "":
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir():
            file_path = output_path / f"{result_idx}.txt"
            with file_path.open("w", encoding="utf-8") as file:
                file.write(output_str + "\n")
    dprint(output_str)
    
    output_json["responses"].append(output_str)


def infer(use_cache, do_sample, warmup, iteration=0):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0 and not warmup:
        dprint(f"use_cache {use_cache};; do_sample {do_sample}")
        dprint("==================")
    if hasattr(model.config, "ntk_scaling") and model.config.ntk_scaling:
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    first = (iteration * args.batch_size)
    if has_padding:
        ids, extra_generation_kwargs = pad_input_ids(prompts[first:first+args.batch_size], min_pad_length=padding_length)
    else:
        ids = prompts
        extra_generation_kwargs = None
    
    # Add only_last_token optimization
    # global extra_generation_kwargs
            
    if extra_generation_kwargs is None:
        extra_generation_kwargs = {}
    extra_generation_kwargs["only_last_token"] = True

    if args.device_type == "cpu":
        # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash cpu impl when padding too much
        extra_generation_kwargs["attn_algorithm"] = "math"

    if not args.no_early_termination and not warmup:
        eos_token_id = tokenizer.eos_token_id
    else:
        eos_token_id = None
        
    result = generate(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=use_cache,
        do_sample=do_sample,
        max_seq_len=max_seq_len,
        timing=args.timing,
        eos_token_id=eos_token_id,
        extra_kwargs=extra_generation_kwargs,
    )
    
    if args.timing != "":
        result, timings = result
        if args.timing == "e2e":
            dprint(f"E2E timing information: {timings[0]:.3f}s")
            output_json["e2e_s"].append(timings[0])
        elif args.timing == "per-token":
            total = sum(timings)
            dprint(f"Per-token timing information: {', '.join([f'{t*1000:.3f}' for t in timings])} ms")
            dprint(f"Total timing information: {total:.3f} s to generate {len(timings)} tokens")
            dprint(f"TTFT: {timings[0]*1000:.3f} ms or {timings[0]:.3f} s")
            dprint(f"TPOT: {(sum(timings[1:])*1000)/len(timings[1:]):.3f} ms")
            if not warmup:
                output_json["e2e_s"].append(total)
                output_json["ttft_s"].append(timings[0])
                output_json["itls_s"].append(timings[1:])
                output_json["tpot_ms"].append((sum(timings[1:])*1000)/len(timings[1:]))
                output_json["itl_ms"].append((sum(timings[1:])*1000)/len(timings[1:]))
            else:
                output_json["warmup_e2e_s"].append(total)
                output_json["warmup_ttft_s"].append(timings[0])
                output_json["warmup_itls_s"].append(timings[1:])
                output_json["warmup_tpot_ms"].append((sum(timings[1:])*1000)/len(timings[1:]))
                output_json["warmup_itl_ms"].append((sum(timings[1:])*1000)/len(timings[1:]))
        # if not warmup:
        #     output_json["responses_len"].append(len(timings))
        else:
            output_json["warmup_responses_len"].append(len(timings))
            
    if len(result.shape) == 1:
        result = result.unsqueeze(0)

    if not warmup:
        for i in range(result.shape[0]):
            print_result(result[i], i)
            output_json["responses_len"].append(len(tokenizer.convert_ids_to_tokens(result[i]))-len(ids[i].tolist()))

do_sample = [False]
use_cache = [
    args.no_use_cache
]  # True/False are identical with greedy iff `torch.use_deterministic_algorithms(True)`

if args.compile:
    dprint(f"compilation warmup")
    pt_compile_model_time = time.time()
    for sample, cache in itertools.product(do_sample, use_cache):
        infer(cache, sample, True)
    pt_compile_model_time = time.time() - pt_compile_model_time
    dprint(f"PT compile complete, took {pt_compile_model_time:.3f}s")
    benchmark_json["pt_compile_s"] = pt_compile_model_time

    if is_aiu_backend:
        dprint("executing update_lazyhandle and compiling for AIU")
        update_lh_time = time.time()
        torch_sendnn.update_lazyhandle()
        update_lh_time = time.time() - update_lh_time
        dprint(f"update_lazyhandle complete, took {update_lh_time:.3f}s")
        benchmark_json["update_lazyhandle_s"] = update_lh_time

    if args.device_type == "aiu":  # only run warmup for AIU, no need for senulator
        aiu_warmup_time = time.time()
        for sample, cache in itertools.product(do_sample, use_cache):
            infer(cache, sample, True)
        aiu_warmup_time = time.time() - aiu_warmup_time
        dprint(f"AIU warmup complete, took {aiu_warmup_time:.3f}s")
        benchmark_json["aiu_warmup_s"] = aiu_warmup_time
else:
    benchmark_json["pt_compile_s"] = None
    benchmark_json["update_lazyhandle_s"] = None
    benchmark_json["aiu_warmup_s"] = None
    
dprint(f"generating output")

inference_start_time = time.perf_counter()

for sample, cache in itertools.product(do_sample, use_cache):
    for _ in range(args.iters):
        infer(cache, sample, False, _)

inference_duration = time.perf_counter() - inference_start_time

benchmark_duration = time.perf_counter() - benchmark_start_time


selected_percentiles = [25,50,75,99]
        
result = {
        "duration": benchmark_duration,
        "completed": len(output_json["responses_len"]),
        "total_input_tokens": sum(output_json["prompts_len"]),
        "total_output_tokens": sum(output_json["responses_len"]),
        "request_throughput": len(output_json["responses_len"]) / inference_duration,
        "request_goodput:": None,
        "output_throughput": sum(output_json["responses_len"]) / inference_duration,
        "total_token_throughput": (sum(output_json["prompts_len"])+sum(output_json["responses_len"]))/ inference_duration,
        "input_lens": output_json["prompts_len"],
        "output_lens": output_json["responses_len"],
        "ttfts": output_json["ttft_s"],
        "itls": output_json["itls_s"],
        "generated_texts": output_json["responses"],
        "errors": [],
        "mean_ttft_ms": np.mean(output_json["ttft_s"] or 0)*1000,  # ttfts is empty if streaming is not supported by backend
        "std_ttft_ms": np.std(output_json["ttft_s"] or 0)*1000,
        "median_ttft_ms": np.median(output_json["ttft_s"] or 0)*1000 ,
    }

for p in selected_percentiles:
    result[f"p{p}_ttft_ms"] = np.percentile(output_json["ttft_s"] or 0, p) *1000
    
result["mean_tpot_ms"] = np.mean(output_json["tpot_ms"] or 0)  # ttfts is empty if streaming is not supported by backend
result["std_tpot_ms"] = np.std(output_json["tpot_ms"] or 0)
result["median_tpot_ms"] = np.median(output_json["tpot_ms"] or 0)
for p in selected_percentiles:
    result[f"p{p}_tpot_ms"] = np.percentile(output_json["tpot_ms"] or 0, p)
    
result["mean_itl_ms"] = np.mean(output_json["itl_ms"] or 0)  # ttfts is empty if streaming is not supported by backend
result["std_itl_ms"] = np.std(output_json["itl_ms"] or 0)
result["median_itl_ms"] = np.median(output_json["itl_ms"] or 0)
for p in selected_percentiles:
    result[f"p{p}_itl_ms"] = np.percentile(output_json["itl_ms"] or 0, p)

result["mean_e2el_ms"] = np.mean(output_json["e2e_s"] or 0)*1000 # ttfts is empty if streaming is not supported by backend
result["std_e2el_ms"] = np.std(output_json["e2e_s"] or 0)*1000
result["median_e2el_ms"] = np.median(output_json["e2e_s"] or 0)*1000
for p in selected_percentiles:
    result[f"p{p}_e2el_ms"] = np.percentile(output_json["e2e_s"] or 0, p)*1000

# # Setup
current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
vllm_json = {}
vllm_json["date"] = current_dt
vllm_json["backend"] = os.getenv("COMPONENT_NAME", "fms")
vllm_json["model_id"] = output_json["model_path"].rstrip("/").split("/")[-1] if output_json["model_path"] else output_json["variant"].rstrip("/").split("/")[-1] 
vllm_json["tokenizer_id"] = output_json["tokenizer"].rstrip("/")
vllm_json["best_of"] = None
vllm_json["num_prompts"] = len(output_json["prompts"])
vllm_json["request_rate"] = result["completed"] / result["duration"] #review
vllm_json["burstiness"] = None
vllm_json["max_concurrency"] = None

# # Merge with benchmark result
result_json = {**vllm_json, **result}

# Compute throughput using: Throughput=Batch Size/Inter-Token Latency (ITL)T
benchmark_json["batch_output_token_throughput"] = args.batch_size / result["mean_itl_ms"]

benchmark_json["inference_duration_s"] = inference_duration

columns = ["pwr", "gtemp", "rdmem", "wrmem", "rxpci", "txpci",  "rdrdma", "wrrdma" ]
metrics = [ "min", "mean", "std", "median", "max" ]

power_json = { }
for c in columns:
    for state in [ "idle", "busy" ]:
        for m in metrics:
            power_json[f"{c}_{state}_{m}"] = None

benchmark_json["prompts"] = output_json["prompts"]

if args.json_output_file != "":
    
    try:
        import pandas as pd
        # Replace 'yourfile.txt' with the path to your .txt file
        log_file = args.json_output_file.replace(".json", "_power.log")

        # Read the text file into a DataFrame, specifying the delimiter as '\t' (tab)
        df = pd.read_csv(log_file, sep='\s+', header=0)
        df = df.drop(index=0)
        mapping_dict = {'-': None}
        df = df.replace(mapping_dict)

        for c in df.columns[2:].to_list():
            df[c] = df[c].astype(float)
            
        import numpy as np
        
        for c in columns:
            power_json[f"{c}_idle_min"] = np.min(df[(df["busy"]==0)][c])
            power_json[f"{c}_idle_mean"] = np.mean(df[(df["busy"]==0)][c])
            power_json[f"{c}_idle_std"] = np.std(df[(df["busy"]==0)][c])
            power_json[f"{c}_idle_median"] = np.median(df[(df["busy"]==0)][c])
            power_json[f"{c}_idle_max"] = np.max(df[(df["busy"]==0)][c])
            power_json[f"{c}_busy_min"] = np.min(df[(df["busy"]>1)][c])
            power_json[f"{c}_busy_mean"] = np.mean(df[(df["busy"]>1)][c])
            power_json[f"{c}_busy_std"] = np.std(df[(df["busy"]>1)][c])
            power_json[f"{c}_busy_median"] = np.median(df[(df["busy"]>1)][c])
            power_json[f"{c}_busy_max"] = np.max(df[(df["busy"]>1)][c])
    except:
        print(f"Unable to parse power file {log_file}")
        pass

    result_json = {**result_json, **benchmark_json, **power_json}

    with open(args.json_output_file, "w", encoding="utf-8") as file:
        json.dump(result_json, file, indent=2)
import argparse
import json
import os
from pathlib import Path
import typing
import uuid

import torch
import transformers
from compressed_tensors.quantization.quant_args import QuantizationArgs
from compressed_tensors.quantization.quant_args import QuantizationType
from compressed_tensors.quantization.quant_args import QuantizationStrategy

import pfgen


class Callback:
    def __init__(self) -> None:
        self.tokenizer: transformers.PreTrainedTokenizer | None = None
        self.model: transformers.PreTrainedModel | None = None
        self.counter = 0
        self.uuid = str(uuid.uuid4())

    def get_kv_cache_filename(self):
        filename = Path(f"out/kv_cache/{self.uuid}/{self.counter}")
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.counter += 1
        return filename
 
    def __call__(
        self, tasks: list[dict[str, str]], params: dict[str, typing.Any]
    ) -> typing.Iterator[str | None]:
        model_id = params.get("_path", None) or params["model"]
        mode = params["mode"]
        if self.model is None:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_id, padding_side="left", trust_remote_code=True
            )
            self.tokenizer.add_eos_token = False
            model_kwargs = {}
            if params.get("dtype", None):
                model_kwargs["torch_dtype"] = params["dtype"]
            device = params.get("_device", "cpu")
            if device == "auto":
                model_kwargs["device_map"] = "auto"
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, **model_kwargs
            )
            self.model.eval()
            if device != "auto":
                self.model.to(device)
            if params.get("quant_args", None):
                self.model.config.init_quant_args(
                    num_bits=args.qbits,
                    type=args.qtype,
                    group_size=args.qgroup,
                    strategy=args.qstrategy,
                    verbose=args.verbose,
                    scale_and_zp_file=args.scale_and_zp,
                )
        assert self.tokenizer is not None
        tokenizer: transformers.PreTrainedTokenizer = self.tokenizer
        if not hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = tokenizer.eos_token
        if params.get("chat_template", None):
            tokenizer.chat_template = params["chat_template"]
        assert self.model is not None
        model: transformers.PreTrainedModel = self.model
        if not hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id

        task_groups: list[list[dict[str, str]]] = [[]]
        for task in tasks:
            if len(task_groups[-1]) >= params["_batch_size"]:
                task_groups.append([])
            task_groups[-1].append(task)

        for task_group in task_groups:
            if mode == "completion":
                inputs = tokenizer(
                    [t["prompt"] for t in task_group], return_tensors="pt", padding=True
                )
            elif mode == "chat" or mode == "qa":
                chats = []
                for task in task_group:
                    if "system_prompt" in task:
                        chat = [
                            {"role": "system", "content": task["system_prompt"]},
                            {"role": "user", "content": task["user_prompt"]},
                        ]
                    else:
                        chat = [{"role": "user", "content": task["prompt"]}]
                    chats.append(chat)
                inputs = tokenizer.apply_chat_template(
                    conversation=chats,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            try:
                # NOTE: Workaround for models such as Tanuki-8x8B-dpo-v1.0.
                if "token_type_ids" in inputs:
                    del inputs["token_type_ids"]
                stop_strings = params.get("stop", []).copy()
                if tokenizer.eos_token is not None:
                    stop_strings.append(tokenizer.eos_token)
                if tokenizer.bos_token is not None:
                    stop_strings.append(tokenizer.bos_token)
                torch.manual_seed(task.get("seed", 0))
                do_sample = params["temperature"] > 1e-6
                output_with_cache = model.generate(
                    **{k: v.to(model.device) for k, v in inputs.items()},
                    max_new_tokens=params.get("max_tokens", 300),
                    do_sample=do_sample,
                    temperature=params["temperature"] if do_sample else None,
                    top_p=params["top_p"] if do_sample else None,
                    top_k=None,
                    pad_token_id=tokenizer.eos_token_id,
                    tokenizer=tokenizer,
                    stop_strings=stop_strings,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=False,  # Optional: Set to False to reduce output size if not needed
                    output_hidden_states=False,  # Optional: Set to False to reduce output size if not needed
                )
                outputs = output_with_cache.sequences
                if params.get("save_kv_cache", False):
                    kv_cache = output_with_cache.past_key_values
                    torch.save(kv_cache, self.get_kv_cache_filename())
            except Exception as e:
                raise e
                for _ in task_group:
                    yield None
                continue
            for output in outputs:
                result = tokenizer.decode(
                    output[inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                for stop in params.get("stop", []):
                    if result.endswith(stop):
                        result = result[: -len(stop)]
                yield result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--mode",
        type=str,
        default="completion",
        choices=["chat", "qa", "completion"],
        help="Which chat template to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Huggingface model name.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the model.",
    )
    parser.add_argument("--num-trials", type=int, default=10, help="Number of trials to run.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")
    parser.add_argument("--top-p", type=float, default=0.98, help="Top-p for sampling.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for sampling.")
    parser.add_argument("--device", type=str, default="auto", help="Device for sampling.")
    parser.add_argument("--dtype", type=str, default="", help="Data type.")
    parser.add_argument(
        "--qbits",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--qtype",
        type=str,
        default="float",
        choices=["float", "int"],
    )
    parser.add_argument(
        "--qstrategy",
        type=str,
        default="static",
        choices=["static", "token", "channel"],
    )
    parser.add_argument(
        "--qgroup",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--scale_and_zp",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--save_kv_cache",
        action="store_true",
    )
    args = parser.parse_args()
    kwargs = {}
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.mode != "completion" and os.path.exists("chat_templates.json"):
        with open("chat_templates.json") as f:
            chat_templates = json.load(f)
            for t in chat_templates:
                if args.model in t["models"]:
                    kwargs["chat_template"] = t["chat_template"]
    if args.qbits:
        kwargs["quant_args"] = {
            "num_bits": args.qbits,
            "type": args.qtype,
            "strategy": args.qstrategy,
            "group_size": args.qgroup,
            "scale_and_zp_file": args.scale_and_zp,
            "verbose": args.verbose,
        }
    kwargs["save_kv_cache"] = args.save_kv_cache
    pfgen.run_tasks(
        args.mode,
        Callback(),
        engine="hf",
        model=args.model,
        num_trials=args.num_trials,
        temperature=args.temperature,
        top_p=args.top_p,
        _path=args.path,
        _batch_size=args.batch_size,
        _device=args.device if torch.cuda.is_available() else "cpu",
        **kwargs,
    )

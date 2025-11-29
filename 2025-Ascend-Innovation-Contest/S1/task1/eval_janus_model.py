from pathlib import Path
import time
import numpy as np
import os
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any
import mindspore
import mindspore as ms
from mindnlp.transformers import AutoModelForCausalLM
from mindnlp.core import no_grad
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


def array_softmax_close(a, b, rtol=1e-2, atol=1e-2):
    if not isinstance(a, ms.Tensor):
        a = ms.Tensor(a)
    if not isinstance(b, ms.Tensor):
        b = ms.Tensor(b)
    a = ms.mint.softmax(a, dim=-1, dtype=ms.float32)
    b = ms.mint.softmax(b, dim=-1, dtype=ms.float32)
    return ms.mint.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


class JanusInferenceEvaluator:
    def __init__(self, model_path: str, model_name: str, output_dir):

        self.model_name = model_name
        self.output_dir = output_dir
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
            model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, ms_dtype=ms.bfloat16, trust_remote_code=True
        )

        self.model = self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def gen(self, prompt, images, max_new_tokens=1, output_logits=False):

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{prompt}",
                "images": images
            }
        ]

        pil_images = load_pil_images(conversation) if images else []

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        )
        prepare_inputs.pixel_values = prepare_inputs.pixel_values.astype(
            mindspore.bfloat16)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        with no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=output_logits
            )

        return inputs_embeds, outputs

    def evaluate_inference(self, prompts: List[str], image_paths: List[Path], max_new_tokens: int = 50,
                           warm_up_tokens=10,  submit_time=""):
        """
        Evaluate model inference for different prompt lengths.

        :param prompts: List of prompts to evaluate
        :param max_new_tokens: Maximum number of tokens to generate
        :param model_name: Name of the model for file naming
        :param warm_up_tokens: Number of tokens to generate during warm-up
        :return: Detailed evaluation results
        """

        results = {
            'model_name': self.model_name,
            'timestamp': submit_time if submit_time else datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            'prompt_lengths': [],
            'prefill_latencies': [],
            'decode_latencies': [],
            'generated_texts': [],
            'generated_token_num': [],
            'logits_paths': []
        }
        print(image_paths)
        warm_up_prompt = "Hello, this is a warm-up sequence to prepare the Janus model."
        self.gen(warm_up_prompt, ["/home/ma-user/work/task1/demo_images/wash_dog.png"],
                 max_new_tokens=warm_up_tokens, output_logits=False)

        for prompt_id, (prompt, image_path) in enumerate(zip(prompts, image_paths)):
            ms.runtime.synchronize()
            prefill_start = time.time()
            inputs_embeds, model_output = self.gen(
                prompt, [image_path], max_new_tokens=1, output_logits=False)
            ms.runtime.synchronize()
            prefill_end = time.time()
            prefill_latency = prefill_end - prefill_start
            print(f"prefill_latency:{prefill_latency}")

            results['prefill_latencies'].append(prefill_latency)
            prompt_length = inputs_embeds.shape[1]
            results['prompt_lengths'].append(prompt_length)

            ms.runtime.synchronize()
            e2e_latency_start = time.time()
            inputs_embeds, model_output = self.gen(
                prompt, [image_path], max_new_tokens=max_new_tokens, output_logits=True)
            ms.runtime.synchronize()
            e2e_latency_end = time.time()
            e2e_latency = e2e_latency_end - e2e_latency_start
            decode_latency = e2e_latency - prefill_latency
            print(
                f"e2e_latency:{e2e_latency}, decode_latency:{decode_latency}")
            mean_decode_latency = decode_latency / \
                (model_output.sequences.shape[1])
            if model_output.logits is not None:
                # 生成唯一的logits文件名：包含模型名、时间戳、prompt_id
                logits_filename = f"{self.model_name}_{results['timestamp']}_prompt_{prompt_id}.npy"
                logits_filepath = os.path.join(
                    self.output_dir, 'logits', logits_filename)
                os.makedirs(os.path.dirname(logits_filepath), exist_ok=True)

                # 将logits转换为numpy数组并保存为.npy文件
                logits_list = [step_logits.asnumpy()
                               for step_logits in model_output.logits]
                logits_np = np.stack(logits_list, axis=0)  # 按step维度堆叠
                np.save(logits_filepath, logits_np)
                results['logits_paths'].append(logits_filepath)
            results['generated_token_num'].append(
                model_output.sequences.shape[1])
            results['decode_latencies'].append(mean_decode_latency)
            results['generated_texts'].append(
                self.tokenizer.decode(model_output.sequences[0], skip_special_tokens=True))
            mindspore.runtime.empty_cache()

        results['overall_avg_latencies'] = {
            'avg_prefill_latency': np.mean(results['prefill_latencies']),
            'avg_decode_latency': np.mean(results['decode_latencies'])
        }
        results['memory_allocated'] = mindspore.runtime.max_memory_allocated() / \
            10**9
        results['memory_reserved'] = mindspore.runtime.max_memory_reserved() / \
            10**9
        return results

    def summarize_results(self, results: Dict[str, Any], target_results) -> None:
        os.makedirs(os.path.join(self.output_dir, 'results'), exist_ok=True)
        all_same = True  # 总结果标记
        check_log = ""
        compare_results = []  # 存储每个prompt的详细比较结果

        # 遍历每个prompt进行比较
        for i in range(len(results['generated_texts'])):
            prompt_result = {
                'prompt_id': i,
                'text_match': False,
                'logits_close': False
            }

            # 1. 文本内容比较
            cur_text = results['generated_texts'][i]
            target_text = target_results['generated_texts'][i]

            # 标准化文本比较（忽略首尾空格和大小写）
            if cur_text.strip().lower() == target_text.strip().lower():
                prompt_result['text_match'] = True
                prompt_result['logits_close'] = True
            else:
                # 2. Logits数值比较（仅当文本不匹配时执行）
                try:
                    # 加载当前logits
                    cur_logits_path = results['logits_paths'][i]
                    cur_logits = np.load(cur_logits_path)

                    # 加载目标logits
                    target_logits_path = target_results['logits_files'][i]
                    target_logits = np.load(target_logits_path)

                    # 检查形状一致性
                    if cur_logits.shape != target_logits.shape:
                        check_log += (f"Prompt {i} logits shape mismatch: "
                                      f"Generated {cur_logits.shape} vs Target {target_logits.shape}\n")
                        prompt_result['logits_close'] = False
                    else:
                        is_close = array_softmax_close(
                            cur_logits, target_logits, rtol=1e-3, atol=1e-3)
                        print(f'is_close:{is_close}')
                        # 判断logits是否匹配（阈值可配置）
                        if is_close:
                            prompt_result['logits_close'] = True
                        else:
                            check_log += (f"Prompt {i} logits mismatch\n")
                except Exception as e:
                    check_log += f"Prompt {i} logits comparison failed: {str(e)}\n"
                    prompt_result['logits_close'] = False

            # 更新总结果标记
            if not prompt_result['logits_close']:
                all_same = False
                if not prompt_result['text_match']:
                    check_log += (f"Prompt {i} text mismatch:\n"
                                  f"  Generated: {cur_text}\n"
                                  f"  Target: {target_text}\n")

            compare_results.append(prompt_result)

        # 存储比较结果
        results['all_same'] = all_same
        results['check_log'] = check_log
        results['compare_results'] = compare_results
        results['match_count'] = sum(
            1 for r in compare_results if r['logits_close'])
        results['total_count'] = len(compare_results)

        # 保存结果文件
        results_filename = f'results/result-{results["model_name"]}-{results["timestamp"]}.json'
        results_filepath = os.path.join(self.output_dir, results_filename)
        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # 控制台输出结果摘要
        print("="*20+"Results Summary"+"="*20)
        print(f"Results saved to {results_filepath}")
        print(f"ALL_SAME: {all_same}")
        print(
            f"MATCH_COUNT: {results['match_count']}/{results['total_count']}")
        print(
            f"AVG_PREFILL_LATENCY: {results['overall_avg_latencies']['avg_prefill_latency']:.4f}s")
        print(
            f"AVG_DECODE_LATENCY: {results['overall_avg_latencies']['avg_decode_latency']:.4f}s/token")

        # 输出详细比较结果
        print("\nDetailed Comparison:")
        for comp in compare_results:
            status = "MATCH" if comp['logits_close'] else "MISMATCH"
            details = f"Prompt {comp['prompt_id']}: {status}"
            if comp['text_match']:
                details += " (by text)"
            elif comp['logits_close']:
                details += " (by logits)"
            print(details)


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Evaluate Janus model inference')
    parser.add_argument('--model-path', type=str,
                        help='Path to the Janus model')
    parser.add_argument('--group-name', type=str,
                        default='TestGroup', help='Group Name')
    parser.add_argument('--target-results', type=str,
                        help='target_results.json file path')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--datetime', type=str, default='')
    parser.add_argument('--prompts', type=str, nargs='+', default=["Describe the image",
                                                                   "Generate a Story about the image", ],
                        help='Prompts to evaluate')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--warm-up-tokens', type=int,
                        default=10, help='Number of tokens for warm-up')
    parser.add_argument('--image-paths', type=str, nargs='+',
                        default=['~/work/task1/demo_images/cat.png',
                                 '~/work/task1/demo_images/man.png'],
                        help='Paths to images (one per prompt, required)')
    args = parser.parse_args()
    print(args)
    model_name_prefix = args.model_path.strip(
        '/').split('/')[-1]  # e.g., Qwen2.5-1.5B-Instruct
    model_name = "_".join([model_name_prefix, args.group_name]) # e.g., Qwen2.5-1.5B-Instruct_username
    with open(args.target_results, 'r', encoding='utf-8') as f:
        target_results = json.load(f)
    target_results = target_results[model_name_prefix]
    evaluator = JanusInferenceEvaluator(
        args.model_path, model_name, args.output_dir)
    results = evaluator.evaluate_inference(
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        warm_up_tokens=args.warm_up_tokens,
        image_paths=args.image_paths,
        submit_time=args.datetime
    )
    evaluator.summarize_results(results, target_results)

import time
import numpy as np
import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import mindspore
import mindspore as ms
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.core import no_grad


def array_softmax_close(a, b, rtol=1e-2, atol=1e-2):
    if not isinstance(a, ms.Tensor):
        a = ms.Tensor(a)
    if not isinstance(b, ms.Tensor):
        b = ms.Tensor(b)
    a = ms.mint.softmax(a, dim=-1, dtype=ms.float32)
    b = ms.mint.softmax(b, dim=-1, dtype=ms.float32)
    return ms.mint.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


class LMInferenceEvaluator:
    def __init__(self, model_path, model_name, output_dir):

        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False if 'Qwen3' in model_name else True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, ms_dtype=ms.bfloat16,
        )

        self.model = self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    def gen(self, prompt, max_new_token=1, output_logits=False):

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="ms")
        with no_grad():
            outputs = self.model.generate(
                **model_inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=max_new_token,
                return_dict_in_generate=True,
                output_logits=output_logits
            )

        return model_inputs, outputs

    def evaluate_inference(self, prompts: List[str], max_new_tokens: int = 50, submit_time=""
                           ) -> Dict[str, Any]:

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

        self.gen("Hello, this is a warm-up sequence to prepare the model.",
                 10, output_logits=False)

        for prompt_id, prompt in enumerate(prompts):
            print("=" * 20 + 'Generating' + "=" * 20)
            print("Prompt : {}".format(prompt))
            ms.runtime.synchronize()
            # prefill阶段耗时
            prefill_start = time.time()
            model_inputs, model_output = self.gen(prompt, 1)
            ms.runtime.synchronize()
            prefill_end = time.time()
            prefill_latency = prefill_end - prefill_start
            print(f"prefill_latency:{prefill_latency}")
            results['prefill_latencies'].append(prefill_latency)
            prompt_length = model_inputs['input_ids'].shape[1]
            results['prompt_lengths'].append(prompt_length)
            ms.runtime.synchronize()
            # prefill+decode阶段耗时
            e2e_latency_start = time.time()
            model_input, model_output = self.gen(
                prompt, max_new_tokens, output_logits=True)
            ms.runtime.synchronize()
            e2e_latency_end = time.time()
            e2e_latency = e2e_latency_end - e2e_latency_start
            decode_latency = e2e_latency - prefill_latency
            print(
                f"e2e_latency:{e2e_latency}, decode_latency:{decode_latency}")
            mean_decode_latency = decode_latency / \
                (model_output.sequences.shape[1] -
                 model_input['input_ids'].shape[1])

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
                model_output.sequences.shape[1] - model_input['input_ids'].shape[1])
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
        description='Evaluate LM inference')
    parser.add_argument('--model-path', type=str,
                        default='llm/Qwen1.5-MoE-A2.7B-Chat', help='Path to the model')
    parser.add_argument('--group-name', type=str,
                        default='TestGroup', help='submitter group name')
    parser.add_argument('--output-dir', default="eval_output", type=str)
    parser.add_argument('--datetime', type=str, default='')
    parser.add_argument('--target-results', type=str,
                        default='target_results.json',
                        help='target_results.json file path')
    parser.add_argument('--prompts', type=str, nargs='+', default=["Hello, how are you?",
                                                                   "This American studied art at Yale and is the author of multiple popular mystery novels. First name is 'Hillary'. What's the last name?",
                                                                   """Summarize the following text: US President Donald Trump has said he is 'not happy' with his Russian counterpart Vladimir Putin, following Moscow's largest aerial attack yet on Ukraine.
In a rare rebuke, Trump said: "What the hell happened to him? He's killing a lot of people." He later called Putin "absolutely crazy".
Ukrainian President Volodymyr Zelensky earlier said Washington's "silence" over recent Russian attacks was encouraging Putin, urging "strong pressure" - including tougher sanctions - on Moscow.
"""
                                                                   ], help='Prompts to evaluate')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                        help='Maximum number of tokens to generate')

    args = parser.parse_args()
    print(args)
    model_name_prefix = args.model_path.strip(
        '/').split('/')[-1]  # e.g., Qwen2.5-1.5B-Instruct
    # e.g., Qwen2.5-1.5B-Instruct_username
    model_name = "_".join([model_name_prefix, args.group_name])
    with open(args.target_results, 'r', encoding='utf-8') as f:
        target_results = json.load(f)
    target_results = target_results[model_name_prefix]
    print(f"Evaluating model: {model_name}")
    evaluator = LMInferenceEvaluator(
        args.model_path, model_name=model_name, output_dir=args.output_dir)
    results = evaluator.evaluate_inference(
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        submit_time=args.datetime
    )

    evaluator.summarize_results(results, target_results=target_results)

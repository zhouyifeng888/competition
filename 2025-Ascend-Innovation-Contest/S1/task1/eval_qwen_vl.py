import time
import numpy as np
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import mindspore
import mindspore as ms
from mindnlp.transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from vl_utils import process_vision_info
ms.set_context(mode=1)

def array_softmax_close(a,b,rtol=1e-2,atol=1e-2):
    if not isinstance(a,ms.Tensor):
        a = ms.Tensor(a)
    if not isinstance(b,ms.Tensor):
        b = ms.Tensor(b)
    a = ms.mint.softmax(a, dim=-1,dtype=ms.float32)
    b = ms.mint.softmax(b, dim=-1,dtype=ms.float32)
    return ms.mint.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)

class QwenVLInferenceEvaluator:
    def __init__(self, model_path: str, model_name: str, output_dir):
        """
        Initialize the Qwen-VL model inference evaluator.

        :param model_path: Path to the pre-trained Qwen-VL model
        :param device: Device to run the model on (cuda/cpu)
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            ms_dtype=ms.bfloat16,

        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model = self.model.eval()

    def generate(self, prompt: str, image_path: str, max_new_tokens: int = 100, output_logits=False):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="ms",
        )
        inputs = inputs

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False,
                                      use_cache=True,
                                      return_dict_in_generate=True,
                                      output_logits=output_logits
                                      )
        return inputs.input_ids, outputs

    def evaluate_inference(
            self,
            prompts: List[str],
            image_paths: List[str],
            max_new_tokens: int = 50,
            warm_up_tokens: int = 10,
            submit_time=""
    ) -> Dict[str, Any]:

        results = {
            'model_name': self.model_name,
            'timestamp': submit_time if submit_time else datetime.now().strftime('%Y-%m-%d_%H-%M-%S') ,
            'prompt_lengths': [],
            'prefill_latencies': [],
            'decode_latencies': [],
            'generated_texts': [],
            'generated_token_num': [],
            'logits_paths':[]
        }
        print(image_paths)
        for _ in range(5):
            self.generate("This a warm up prompt",
                          "/home/ma-user/work/task1/demo_images/wash_dog.png", max_new_tokens=warm_up_tokens)

        for prompt_id, (prompt, image_path) in enumerate(zip(prompts, image_paths)):
            # Measure prefill stage latency
            ms.runtime.synchronize()
            prefill_start = time.time()
            inputs_embeds, model_output = self.generate(
                prompt, image_path, max_new_tokens=1)
            ms.runtime.synchronize()
            prefill_end = time.time()
            prefill_latency = prefill_end - prefill_start
            print(f"prefill_latency:{prefill_latency}")

            results['prefill_latencies'].append(prefill_latency)
            prompt_length = inputs_embeds.shape[1]
            results['prompt_lengths'].append(prompt_length)

            # Measure decode stage latency
            ms.runtime.synchronize()
            e2e_latency_start = time.time()
            input_ids, model_output = self.generate(
                prompt, image_path, max_new_tokens, output_logits=True)
            ms.runtime.synchronize()
            e2e_latency_end = time.time()
            e2e_latency = e2e_latency_end - e2e_latency_start
            decode_latency = e2e_latency
            print(
                f"e2e_latency:{e2e_latency}, decode_latency:{decode_latency}")
            print(f"model_output.sequences.shape[1]:{model_output.sequences.shape[1]}")
            print(f"input_ids.shape[1]:{input_ids.shape[1]}")
            mean_decode_latency = decode_latency / \
                                  (model_output.sequences.shape[1] - input_ids.shape[1])
            if model_output.logits is not None:
                # 生成唯一的logits文件名：包含模型名、时间戳、prompt_id
                logits_filename = f"{self.model_name}_{results['timestamp']}_prompt_{prompt_id}.npy"
                logits_filepath = os.path.join(self.output_dir, 'logits', logits_filename)
                os.makedirs(os.path.dirname(logits_filepath), exist_ok=True)

                # 将logits转换为numpy数组并保存为.npy文件
                logits_list = [step_logits.asnumpy() for step_logits in model_output.logits]
                logits_np = np.stack(logits_list, axis=0)  # 按step维度堆叠
                np.save(logits_filepath, logits_np)
                results['logits_paths'].append(logits_filepath)
            results['generated_token_num'].append(model_output.sequences.shape[1] - input_ids.shape[1])
            results['decode_latencies'].append(mean_decode_latency)
            results['generated_texts'].append(
                self.processor.decode(model_output.sequences[0, input_ids.shape[1]:]))
            mindspore.runtime.empty_cache()

        # Calculate and store overall average latencies
        results['overall_avg_latencies'] = {
            'avg_prefill_latency': np.mean(results['prefill_latencies']),
            'avg_decode_latency': np.mean(results['decode_latencies'])
        }
        results['memory_allocated'] = mindspore.runtime.max_memory_allocated()/10**9
        results['memory_reserved'] = mindspore.runtime.max_memory_reserved()/10**9
        
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
                        is_close = array_softmax_close(cur_logits, target_logits, rtol=1e-3, atol=1e-3)
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
        results['match_count'] = sum(1 for r in compare_results if r['logits_close'])
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
        print(f"MATCH_COUNT: {results['match_count']}/{results['total_count']}")
        print(f"AVG_PREFILL_LATENCY: {results['overall_avg_latencies']['avg_prefill_latency']:.4f}s")
        print(f"AVG_DECODE_LATENCY: {results['overall_avg_latencies']['avg_decode_latency']:.4f}s/token")
        
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


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate Qwen-VL model inference')
    parser.add_argument('--model-path', type=str,
                        default='llm/Qwen2-VL-2B-Instruct',
                        help='Path to the Qwen-VL model')
    parser.add_argument('--group-name', type=str,
                        default='TestGroup', help='submitter group name')
    parser.add_argument('--output-dir', default='eval_output', type=str)
    parser.add_argument('--datetime', type=str, default='')
    parser.add_argument('--target-results', type=str,
                        default='target_results.json',
                        help='target_results.json file path')
    parser.add_argument('--prompts', type=str, nargs='+',
                        default=["Describe the image",
                                 "Generate a Story about the image", ],
                        help='Prompts to evaluate')    
    parser.add_argument('--image-paths', type=str, nargs='+',
                        default=['demo_images/cat.png',
                                 'demo_images/man.png'],
                        help='Paths to images (one per prompt, required)')
    parser.add_argument('--max-new-tokens', type=int, default=50,
                        help='Maximum number of tokens to generate per prompt')

    args = parser.parse_args()
    model_name_prefix = args.model_path.strip('/').split('/')[-1]  # e.g., Qwen2.5-1.5B-Instruct
    model_name = "_".join([model_name_prefix, args.group_name])  # e.g., Qwen2.5-1.5B-Instruct_username
    with open(args.target_results, 'r', encoding='utf-8') as f:
        target_results = json.load(f)
    target_results = target_results[model_name_prefix]
    evaluator = QwenVLInferenceEvaluator(
        model_path=args.model_path,
        model_name=model_name,
        output_dir=args.output_dir
    )

    # Run evaluation
    results = evaluator.evaluate_inference(
        prompts=args.prompts,
        image_paths=args.image_paths if args.image_paths else None,
        max_new_tokens=args.max_new_tokens,
        submit_time=args.datetime
    )
    # Print and save summary
    evaluator.summarize_results(results, target_results=target_results)


if __name__ == '__main__':
    main()


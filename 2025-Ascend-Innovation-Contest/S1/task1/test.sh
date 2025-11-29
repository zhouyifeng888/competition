unset RANK_TABLE_FILE
taskset -c 0-23  python eval_dense_moe_model.py  --model-path llm/deepseek-moe-16b-chat

taskset -c 0-23  python eval_dense_moe_model.py  --model-path llm/Qwen1.5-MoE-A2.7B-Chat

taskset -c 0-23  python eval_qwen_vl.py  --model-path llm/Qwen2-VL-2B-Instruct

TASK_DIR=`pwd`
cp eval_janus_model.py  mindnlp/llm/inference/janus_pro/
cd mindnlp/llm/inference/janus_pro/
taskset -c 0-23 python eval_janus_model.py   --model-path ${TASK_DIR}/llm/Janus-Pro-7B --target-results ${TASK_DIR}/target_results.json --image-paths "${TASK_DIR}/demo_images/cat.png" "${TASK_DIR}/demo_images/man.png" --output-dir  ${TASK_DIR}/eval_output/
cd ${TASK_DIR}
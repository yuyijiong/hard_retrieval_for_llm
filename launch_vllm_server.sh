CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m vllm.entrypoints.openai.api_server\
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 5000 \
    --served-model-name llama3.1-70b \
    --model Meta-Llama-3_1-70B-Instruct \
  --max-model-len 50000 >vllm_llama3.1-70b.log 2>&1 &

OPENAI_API_KEY="xxx" \
swift sample \
    --sampler_type distill \
    --sampler_engine client \
    --model deepseek-r1 \
    --stream true \
    --dataset tastelikefeet/competition_math#5 \
    --num_return_sequences 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --engine_kwargs '{"base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1"}'

import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen2.5-3B"
# model_name = "DeepSeek-R1-Distill-Llama-8B"
device = "cuda"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

qwen_id = 151643
deepseek_id = 128001
model_id = qwen_id


user_input = input("user:")

# 开始计时
start_time = time.time()

inputs = tokenizer(f"user:{user_input}\nAI:", return_tensors="pt")
outputs = model.generate(**inputs)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(f"AI:{response}")

# 结束计时
end_time = time.time()
# 计算耗时
elapsed_time = end_time - start_time
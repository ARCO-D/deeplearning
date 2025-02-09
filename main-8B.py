import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "DeepSeek-R1-Distill-Llama-8B"
device = "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# settings
qwen_id = 151643
deepseek_id = 128001
model_id = deepseek_id
max_new_tokens = 200 # 最大输出长度
max_history_length = 10


def chat():
    print("Dialog starts! input 'exit' to end dialog")
    print("[1] Add '[debug]'at the end of input so that you can see the full response of AI")
    print("[2] Inputting 'history' can show last max_history_length of dialog history")
    history = []
    full_response = False

    while True:
        user_input = input("user:")
        if user_input.lower() == "exit":
            print("对话结束")
            break

        if user_input.find("[debug]") != -1:
            user_input = user_input[:user_input.find("[debug]")]
            full_response = True

        if user_input.lower() == "history":
            for i in range(0, len(history)):
                print(f"his-{i}: {history[i]}")
            continue

        # 将用户输入添加到历史记录中
        history.append(f"user:{user_input}")

        # 手动构建上下文
        context = "\n".join(history) + "\nAI:"

        print(f"context: {context}\n")

        # 将上下文编码为模型输入
        inputs = tokenizer(context, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids']
        inputs['attention_mask'] = inputs['attention_mask']

        # 开始计时
        start_time = time.time()

        # 生成模型的回复
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,  # 控制生成的新 token 数量
            pad_token_id=model_id,
            eos_token_id=model_id,
            num_return_sequences=1,
            # do_sample=True,  # 启用采样
            # top_k=50,  # 使用 top-k 采样
            # top_p=0.95  # 使用 top-p 采样
        )

        # 结束计时
        end_time = time.time()
        # 计算耗时
        elapsed_time = end_time - start_time

        # 解码模型的回复
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 截断回复，阻止模型自问自答
        index_start = response.find("AI:")
        index_end = response.find("user:")
        if index_start >= index_end:
            index_start = 0
        if index_start > 0:
            index_start += len("AI:")
        expect_response = response[index_start:index_end]

        history.append(f"AI:{expect_response}")

        # 打印 AI 的回复
        if full_response:
            print(f"full_response:{response}")
            full_response = False
        print(f"AI:{expect_response}")

        # 打印耗时
        print(f"本轮对话耗时: {elapsed_time:.2f} 秒")

        # 控制对话历史长度，避免过长
        if len(history) > max_history_length:
            history = history[-max_history_length:]


if __name__ == "__main__":
    chat()

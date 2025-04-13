from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer, TextStreamer
import torch
import time


max_memory = {
    0 : "12GB",
    1 : "12GB",
    "cpu": "48GB"
}
model_path = "/data2/models/vlm/Kimi-VL-A3B-Thinking"
model = None

# 1. 加载模型和处理器
def load_model():
    global model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True
    )

# 2. AI处理函数（支持流式输出）
def ai_deal(context):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    start_time = time.time()

    # 处理输入消息（参考[[5]]的数据预处理逻辑）
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": context}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # 启动流式输出（参考[[2]]的模型生成逻辑）
    # streamer = TextIteratorStreamer(processor, skip_special_tokens=True)
    streamer = TextStreamer(processor, skip_special_tokens=True)
    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 2048,
        "streamer": streamer,
    }


    generated_ids = model.generate(**generate_kwargs)
    #
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    # print(response)

    tokenizer = processor.tokenizer
    tokens = tokenizer.tokenize(response)
    num_tokens = len(tokens)
    print(f"totally {num_tokens} tokens")


def chat():
    context = "你好，请做个自我介绍吧"

    # 开始计时
    start_time = time.time()

    ai_deal(context)

    # 结束计时
    end_time = time.time()
    # 计算耗时
    elapsed_time = end_time - start_time
    # 打印耗时
    print(f"本轮对话耗时: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    load_model()
    chat()
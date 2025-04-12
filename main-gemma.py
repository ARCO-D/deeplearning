from llama_cpp import Llama
import time
import json
import re

## global variable
llm = None
default_sys_prompt = "你是一个AI助手，请用中文回复用户。"

system_prompt = {
    "role": "system",
    "content": {
        "type" : "text",
        "text" : default_sys_prompt
    }
}
user_role = "user"
AI_role = "assistant"

## global settings
# model_path = "/data2/models/llm/others/Llama-4-Scout-17B-16E-Instruct-GGUF/UD-Q3_K_XL/Llama-4-Scout-17B-16E-Instruct-UD-Q3_K_XL.gguf-00001-of-00002.gguf"
# model_path = "/data/deepseek/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q6_K/DeepSeek-R1-Distill-Llama-70B-Q6_K-00001-of-00002.gguf"
# model_path = "/data/models/deepseek/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
# model_path = "/data/models/deepseek/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-IQ4_XS.gguf"
# model_path = "/data/models/deepseek/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
# model_path = "/data2/models/llm/qwen/QwQ-32B-GGUF/qwq-32b-q6_k.gguf"
# model_path = "/data/models/qwen/QwQ-32B-GGUF/qwq-32b-q4_k_m.gguf"
model_path = "/data2/models/llm/others/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q6_K.gguf"
# model_path = "/data2/models/llm/others/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf"
# model_path = "/data/deepseek/DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
# model_path = "/data/deepseek/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
context_length = 10240 # 最大上下文长度
max_tokens = 1024 # AI一次最多生成的tokens
gpu_layers = -1 # 没GPU就填0
threads = 4 # 贴近逻辑核数
truncate_str = "{\"role\": \"user\""

## functions
def help_prompt():
    print("系统提示词，提示AI的身份，例如：你是一位诗人")
    print(f"默认提示词：{default_sys_prompt}")


def load_model():
    global llm
    # 开始计时
    start_time = time.time()
    llm = Llama(model_path=model_path,
                main_gpu=0,
                offload_kqv=True,
                n_ctx=context_length,  # 上下文长度
                n_threads=threads,  # 使用的线程数，根据你的 CPU 核心数进行调整
                n_gpu_layers=gpu_layers)  # 如果你的 GPU 支持，可以设置大于 0 的值来使用 GPU 加速
    # 结束计时
    end_time = time.time()
    # 计算耗时
    elapsed_time = end_time - start_time
    # 打印耗时
    print(f"加载模型耗时耗时: {elapsed_time:.2f} 秒")


def set_system_prompt():
    print("需要设置系统提示词吗 (y/n/help)  ", end="")
    str = input()
    if str == 'y':
        system_prompt["content"] = input("请输入提示词: ")
    elif str == "help":
        help_prompt()
        set_system_prompt()


def ai_deal(context, full_response):
    global llm
    # 开始计时
    start_time = time.time()

    # 生成模型的回复，使用流式生成
    print("AI:", end="")
    response_parts = []
    generator = llm(
            context,
            max_tokens=max_tokens,
            temperature=0.8,
            top_p=0.95,
            repeat_penalty=1.2,    # 惩罚已生成的 token
            frequency_penalty=1.2, # 降低频繁出现 token 的概率
            stream=True,
            stop=truncate_str
    )
    for output in generator:
        token_text = output['choices'][0]['text']
        response_parts.append(token_text)
        if not full_response:
            print(token_text, end='', flush=True)

    # 结束计时
    end_time = time.time()
    # 计算耗时
    elapsed_time = end_time - start_time

    # 处理模型的回复
    response = ''.join(response_parts)
    if truncate_str in response:
        response = response.split(truncate_str)[0]
    expect_response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response)

    # 打印耗时
    print(f"本轮对话耗时: {elapsed_time:.2f} 秒")
    return expect_response


def chat():
    # local settings
    history = []
    full_response = False

    while True:
        user_input = input("user:")
        if user_input.lower() == "exit":
            print("对话结束")
            break

        # 构建用户输入的 JSON 对象字符串并添加到历史记录
        user_msg = {
            "role": f"{user_role}",
            "content": {
                "type" : "text",
                "text" : user_input
            }
        }
        history.append(user_msg)

        # 将history的json数组转换为messages字符串数组
        messages = [json.dumps(system_prompt, ensure_ascii=False)]
        for item in history:
            msg_str = json.dumps(item, ensure_ascii=False)
            messages.append(msg_str)

        # 将messages的字符串数组转换为字符串上下文
        context = "\n".join(messages)
        # print(f"context={context}")

        # 调用处理回复的函数
        expect_response = ai_deal(context, full_response)

        # 构建 AI 回复的 JSON 对象字符串并添加到历史记录
        ai_msg = json.loads(expect_response)
        history.append(ai_msg)

        # 控制对话历史长度，避免过长
        # 通过history + token < content_length判断，暂未实现


if __name__ == "__main__":
    # 加载模型
    load_model()
    # 设置系统提示词
    set_system_prompt()
    # 开始会话
    chat()

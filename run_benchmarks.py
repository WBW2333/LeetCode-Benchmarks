import json
from bot_pipeline import BoT
import argparse
import os
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

import torch
# import time
from datetime import datetime
from human_eval.data import write_jsonl, read_problems
# import torch.nn as nn
from FlagEmbedding import BGEM3FlagModel
import re


class Logger(object):
    def __init__(self, fileN="easy.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


import sys

sys.stdout = Logger("./1.txt")  # 修改参数来确定是覆盖写还是追加写


# model_name = "E:/Work/1/pythonProject/ChatGLM-main/model/ZhipuAI/codegeex2-6b"
# model_name = "E:/Work/1/pythonProject/ChatGLM-main/model/ZhipuAI/codegeex4-all-9b"
# model_name = "E:/Work/1/pythonProject/Qwen-main/model/qwen/CodeQwen1.5-7B-Chat"
model_name = 100


# problems = read_problems('E:/Work/1/pythonProject/humaneval-x/codegeex/benchmark/humaneval-x/python/data/humaneval_python.jsonl.gz')

# 定义读取 JSON 文件的函数
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 读取 JSON 文件
# print("easy")
# file_path = 'E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/leetcode_passk/easy-bench.json'
# print("medium")
# file_path = 'E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/leetcode_passk/medium-bench.json'
print("hard")
file_path = 'E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/leetcode_passk/hard-bench.json'

# print("无")
# print("ATP")
print("全面AT")
# print("小AT")
# print("仅CoT")
# print("仅AT")
problems = read_json_file(file_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 100

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto",
# trust_remote_code=True,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# ).to(device).eval()

model = 100


# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer = 100

# BGE，读取模板
sentences = []
dic = []
# 从JSONL文件逐行读取数据
# with open('E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/thought_templates/thought_templates_small.jsonl', 'r') as f:
# with open('E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/thought_templates/thought_templates.jsonl', 'r') as f:
with open('E:/Work/1/pythonProject/humaneval-x/leetcode-benchmark/thought_templates/output.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        dic.append(data)
        sentences.append(data['thought_question'])

bgeModel = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
output_templates = bgeModel.encode(sentences, return_colbert_vecs=True)
# bgeModel = 100
# output_templates = 100


def get_func_name(text):
    code_match = re.search(r'python\n(.*?)\n', text, re.DOTALL)
    if code_match:
        code = code_match.group(1)
        return code
    return ""

def get_func_define(text):
    function_match = re.search(r'def\s+(\w+)\s*', text)
    if function_match:
        return function_match.group(1)
    return ""


def process_code(code_str):
    # 检查字符串是否以 'class Solution:' 开头
    if code_str.strip().startswith("class Solution:"):
        # 去掉 'class Solution:' 这一行
        lines = code_str.splitlines()
        processed_lines = []
        for line in lines[1:]:  # 跳过第一行
            # 去掉每行开头的一个缩进符（假设缩进为4个空格或一个制表符）
            if line.startswith("    "):  # 如果缩进是4个空格
                processed_lines.append(line[4:])
            elif line.startswith("\t"):  # 如果缩进是一个制表符
                processed_lines.append(line[1:])
            else:
                processed_lines.append(line)

        # 将处理后的代码行重新组合成字符串
        processed_code = "\n".join(processed_lines)
        return processed_code
    else:
        return code_str  # 如果不以 'class Solution:' 开头，返回原始字符串


# 定义生成多个完成的函数
def generate_multiple_completions(task_ids, prompts, num_samples_per_task, start = True, run_start = "", right = 0, wrong = 0):
    results = []
    right = right
    wrong = wrong
    do_run = start
    for task_id, prompt in zip(task_ids, prompts):
        print(datetime.now().strftime("%H:%M:%S"), task_id)
        if do_run == False:
            if task_id == run_start:
                do_run = True
            else:
                continue
        func_name = get_func_name(problems[task_id]['python'])
        func_define = get_func_define(func_name)
        print(f"函数调用：{func_name}，函数定义：{func_define}")
        test_bot = BoT(
            # user_input为输入的问题
            user_input=problems[task_id]['prompt'] + '\n```' + func_name + '\n```' + problems[task_id]['content']['constraints'] + '\n' + problems[task_id]['content']['follow_up'],
            model_name=model_name,
            problems=problems,
            device=device,
            model=model,
            bgeModel=bgeModel,
            dic=dic,
            output_templates=output_templates,
            tokenizer=tokenizer,
            CoT = True,
            ATP = True,
        )
        for _ in range(num_samples_per_task):
            result = test_bot.bot_run()
            print("结果：")
            print(result)
            if result is None:
                continue
            result = process_code(result)
            for input in problems[task_id]['passk']['function_input']:
                code_run = '\nresult = ' + func_define + '(' + input['input'] + ")"
                print(code_run)

                # 定义一个空的字典来存储执行代码后的变量
                exec_globals = {}
                ans = "空结果"
                correct = True
                try:
                    # 使用exec函数执行代码字符串
                    exec (result + code_run, exec_globals)
                    # 获取结果
                    ans = exec_globals['result']
                except Exception as e:
                    # 捕获异常并输出异常信息
                    ans = f"An error occurred: {e}"
                    correct = False

                # 打印结果
                # print(result)
                try:
                    print(f"输出结果：{ans}，样例答案：{input['output']}，监测结果：{ans == eval(input['output'])}")
                    if not ans == eval(input['output']):
                        correct = False
                except Exception as e:
                    continue
            if correct:
                right += 1
            else:
                wrong += 1
            print(f"累计正确：{right}，错误：{wrong}")

            # results.append({
            #     "task_id": task_id,
            #     "generation": result,
            #     "canonical_solution": problems[task_id]["canonical_solution"],
            #     "declaration": problems[task_id]["declaration"],
            #     "example_test": problems[task_id]["example_test"],
            #     "prompt": problems[task_id]["prompt"],
            #     "test": problems[task_id]["test"],
            #     "text": problems[task_id]["text"]
            # })
        # print(test_bot)
    return right/(right+wrong)


if __name__ == "__main__":
    # 生成多条记录
    num_samples_per_task = 1  # 每个任务生成的样本数量
    all_task_ids = list(problems.keys())
    all_prompts = [problems[task_id]["prompt"] for task_id in all_task_ids]

    # 直接生成所有记录
    # samples = generate_multiple_completions(all_task_ids, all_prompts, num_samples_per_task, False, "leetcode-benchmark-77", 51,22)
    samples = generate_multiple_completions(all_task_ids, all_prompts, num_samples_per_task)
    print(samples)
    # write_jsonl("chatCodeQwenPython.jsonl", samples, True)

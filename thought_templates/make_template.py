import json

for path in ["/run/wbw/humaneval-x/leetcode-benchmark/leetcode_passk/easy-bench.json", "/run/wbw/humaneval-x/leetcode-benchmark/leetcode_passk/hard-bench.json", "/run/wbw/humaneval-x/leetcode-benchmark/leetcode_passk/medium-bench.json"]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设我们对每个键值对进行某种修改，例如将所有键加上一个前缀


    # 将修改后的数据转换为JSONL格式并保存
    with open('output.jsonl', 'a', encoding='utf-8') as f:
        for key, value in data.items():
            modified_data = {}
            modified_data["thought_name"] = value['title']
            modified_data["thought_question"] = value['content']['problem']
            modified_data["thought_answer"] = value['python']
            json_line = json.dumps(modified_data, ensure_ascii=False)
            f.write(json_line + '\n')

    print("数据已成功修改并保存为 JSONL 格式文件 'output.jsonl'。")
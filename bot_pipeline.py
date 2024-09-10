# 导入所需的库
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import time
from datetime import datetime
from human_eval.data import write_jsonl, read_problems
import torch.nn as nn
import json
from accelerate import infer_auto_device_map
from meta_buffer_utilis import meta_distiller_prompt,extract_code
import requests



class Pipeline:
    def __init__(self,device,model,tokenizer):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
    def get_respond(self,meta_prompt,user_prompt):
        messages = [
            {"role": "system", "content": meta_prompt},
            {"role": "user", "content": user_prompt},
        ]




        # inputs = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=True, return_tensors="pt", return_dict=True,
        #     add_generation_prompt=True
        # ).to(self.device)
        #
        # outputs = self.model.generate(
        #     **inputs,
        #     max_length=4096,
        #     do_sample=True,
        #     temperature=0.4,
        #     top_p=0.9
        # )
        # outputs = outputs[:, inputs['input_ids'].shape[1]:]
        #
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # print('--------------------------------------------------------------------------------------------------')
        # print(response)

        # return response


        # # print('--------------------------------------------------------------------------------------------------')


        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        # model_inputs = self.tokenizer([text], return_tensors="pt", return_token_type_ids=False).to(self.device)
        #
        # generated_ids = self.model.generate(
        #     model_inputs.input_ids,
        #     max_new_tokens=4096,
        #     do_sample=True,
        #     temperature=0.4,
        #     top_p=0.9
        # )
        # generated_ids = [
        #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        # ]
        #
        # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return response

        url = "https://api.aiskt.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"sk-7MNQ6d4C7YuJDUi1898b39Ab869c40C58bCbE21d418d88D5"
        }
        data = {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "messages": messages
        }

        response = requests.post(url, headers=headers, json=data)
        # print(response)

        if response.status_code == 200:
            if False:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if chunk['choices'][0].get('delta', {}).get('content'):
                            print(chunk['choices'][0]['delta']['content'], end='')
            else:
                return response.json()['choices'][0]['message']['content']
        else:
            return "NULL"

class BoT:
    def __init__(self, user_input,problems,model_name,device,model,tokenizer,bgeModel,output_templates,dic, CoT = False, ATP = False):
        self.model_name = model_name
        self.pipeline = Pipeline(device,model,tokenizer)
        self.user_input = user_input
        self.bgeModel = bgeModel
        self.output_templates = output_templates
        self.dic = dic
        self.CoT = CoT
        self.ATP = ATP
        # Only for test use, stay tuned for our update
        self.problems = problems
    def update_input(self,new_input):
        self.user_input = new_input
    def problem_distillation(self):
        # print(f'User prompt:{self.user_input}')
        self.distilled_information = self.pipeline.get_respond(meta_distiller_prompt, self.user_input)
        # print(f'Distilled information:{self.distilled_information}')

    def buffer_retrieve(self):
        if self.ATP:
            output_1 = self.bgeModel.encode(self.user_input, return_colbert_vecs=True)
            max = torch.tensor(0)
            tmp = 0
            for i in range(len(self.output_templates['colbert_vecs'])):
                colbertScore = self.bgeModel.colbert_score(output_1['colbert_vecs'], self.output_templates['colbert_vecs'][i])
                if max.item() < colbertScore.item():
                    max = colbertScore
                    tmp = i
            self.thought_template = self.dic[tmp]['thought_answer']
        else:
            self.thought_template = ''
        # print(f'Thought_template: {self.thought_template}')
        # print(max)

    def reasoner_instantiation(self):
        # Temporay using selection method to select answer extract method
        self.instantiation_instruct = """
You are a Python expert in problem analysis and can apply previous problem-solving approaches to new issues.  The user will provide a specific task description and a thought template.  Your goal is to analyze the user's task and generate a Python code based on the thought template.  Only provide the code and let the compiler handle it.  
It should be noted that all the python code should be within one code block, the answer should not include more than one code block!  And strictly follow the thought-template to instantiate the python code but you should also adjust the input parameter according to the user input!
        """

        self.formated_input = f"""
Distilled information:
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}

Instantiated Solution:
Please analyze the above user task description and thought template, and generate a specific, detailed solution.  Only provide the Python code.  
        """
        self.inspector_prompt = """
You are an excellent python programming master who are proficient in analyzing and editing python code, and you are also good at understanding the real-world problem. Your task is:
1. Analyze the given python code
2. Edit the input code to make sure the edited code is correct and could run and solve the problem correctly.  
Your respond should follow the format below:
```python
## Edited code here
```
        """
        self.result = self.pipeline.get_respond(self.instantiation_instruct,self.formated_input)
        code_str = extract_code(self.result)
        # print(f'Instantiated reasoning result: {self.result}')
        # print(f'code_str result: {code_str}')
        self.inter_input = f"""
                User_input:{self.user_input}
                {code_str}
                """
        inspector_result = self.pipeline.get_respond(self.inspector_prompt,self.user_input)
        self.final_result = extract_code(inspector_result) if inspector_result.find('```python') != -1 else code_str
        # self.final_result = code_str


    def bot_run(self):
        if self.CoT:
            self.problem_distillation()
            self.buffer_retrieve()
            self.reasoner_instantiation()
        else:
            inspector_result = self.pipeline.get_respond("", self.user_input)
            self.final_result = extract_code(inspector_result) if inspector_result.find('```python') != -1 else inspector_result

        return self.final_result

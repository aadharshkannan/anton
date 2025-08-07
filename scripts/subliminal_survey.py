#!/usr/bin/env python
# coding: utf-8

# # Subliminal Survey Runner

# This notebook loads a survey from a JSON file and uses a Hugging Face model to answer the questions multiple times, reporting response percentages.

# In[1]:


import json
from collections import Counter
from pathlib import Path
from transformers import pipeline


# In[ ]:


# Path to the survey JSON file
survey_path = Path('survey.json')  # update with your own file path

with open(survey_path) as f:
    survey = json.load(f)

preamble = survey.get('preamble', '')
questions = survey['questions']


# In[ ]:


# Initialize model
model_name = 'distilgpt2'
generator = pipeline('text-generation', model=model_name)


# In[ ]:


choice_labels = [chr(ord('A') + i) for i in range(26)]

def ask_question(question):
    q_text = question['question']
    choices = question['choices']
    formatted_choices = '\n'.join([f"{choice_labels[i]}. {c}" for i, c in enumerate(choices)])
    prompt = f"{preamble}\n\nQuestion: {q_text}\nChoices:\n{formatted_choices}\n\nRespond with the letter of your choice."
    output = generator(prompt, max_new_tokens=1, do_sample=True, temperature=0.7)[0]['generated_text'][len(prompt):].strip().upper()
    if output and output[0] in choice_labels[:len(choices)]:
        return choices[choice_labels.index(output[0])]
    for c in choices:
        if c.lower() in output.lower():
            return c
    return None


# In[ ]:


N = 100
results = {q['question']: Counter() for q in questions}

for _ in range(N):
    for q in questions:
        answer = ask_question(q)
        if answer is None:
            results[q['question']]['<no_answer>'] += 1
        else:
            results[q['question']][answer] += 1


# In[ ]:


percentages = {}
for q in questions:
    q_text = q['question']
    total = sum(results[q_text].values())
    percentages[q_text] = {choice: round(count/total*100,2) for choice, count in results[q_text].items()}
percentages


# In[ ]:


# Save results to JSON
import re

def get_op_path():
    prefix = re.sub(r'[^A-Za-z0-9.+-]+', '_', model_name)      # step 1
    prefix = re.sub(r'_+', '_', prefix).strip('_')             # step 2
    
    return Path(f'{prefix}_survey_results.json')

output_path = get_op_path()
with open(output_path, 'w') as f:
    json.dump(percentages, f, indent=2)
print(f'Results saved to {output_path}')


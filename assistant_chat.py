from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList, StoppingCriteria
import json
import torch.cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./Llama-3.2-3B"

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, device_map='auto')

# Define a stop sequence
class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, stop_sequence, tokenizer):
        self.stop_sequence = stop_sequence
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_sequence in decoded_text

stop_sequence = "}"
stop_criteria = StoppingCriteriaList([StopSequenceCriteria(stop_sequence, tokenizer)])

# Function to generate a response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,  # Set a smaller max_length
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=2.0,
        temperature=0.2,
        top_k=5,
        top_p=0.6,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stop_criteria
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text_token = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract the content after the last occurrence
    last_occurrence = output_text.rfind("assistant:\n")
    if last_occurrence != -1:
        output_text_response = output_text[last_occurrence + len("assistant:\n"):]
    while True:
        if output_text_response.endswith('\n'):
            output_text_response = output_text_response[:-1]
        else:
            break
    return output_text_response, output_text_token

record = []
system_prompt = """Your are assistant. You are chatting with your host. You only speak English. You can only respond in one complete sentence and with correct grammar."""


# Chat loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    prompt = ''
    # load system
    prompt += f'role and limit:\n'
    prompt += system_prompt
    prompt += '\n\n'
    # load record
    for i in record:
        prompt += f'{list(i.keys())[0]}:\n'
        prompt += next(iter(i.values()))
        prompt += "\n\n"
    # load input
    prompt += f'user:\n'
    prompt += user_input
    prompt += '\n\n'
    # load begin of output
    prompt += f'assistant:\n'

    
    model_output = generate_response(prompt)
    response, debug = model_output
    
    record.append({"user" : user_input})
    record.append({"assistant" : response})
    print("-----------------------------------------------------------")
    print(f"Assistant: {response}")
    # print("***********************************************************")
    # print(f"Debug: {debug}")
    print("-----------------------------------------------------------")

# with open('array.json', 'w') as json_file:
#     json.dump(record, json_file)
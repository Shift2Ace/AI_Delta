from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import bitsandbytes as bnb
from transformers import LogitsProcessorList, StoppingCriteriaList, StoppingCriteria

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
# Encode input text
input_text = "{'Input': 'what is the result of 1+1','Output':' "
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output with adjusted parameters
outputs = model.generate(
    **inputs,
    max_length=500,  # Set a smaller max_length
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=2.0,
    temperature=0.5,
    top_k=50,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,  # Ensure padding uses the eos token
    stopping_criteria=stop_criteria
)

# Decode and post-process the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("-----------------------------------------------------------")
print(output_text)
print("-----------------------------------------------------------")
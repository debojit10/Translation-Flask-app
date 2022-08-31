from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
import torch
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
with open('input.txt') as f:
    text = f.read()

texts = sent_tokenize(text)

tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)

model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
model = model.to(device)
decoded = ''

# You can also use "translate English to French" and "translate English to Romanian"
for text in texts:
  inputs = tokenizer("translate English to German: "+text, return_tensors="pt").to(device)  # Batch size 1

  outputs = model.generate(input_ids=inputs["input_ids"],
                          #  attention_mask=inputs["attention_mask"],
                           max_length=1024,
                           num_beams=5, 
                           no_repeat_ngram_size=2, 
                           early_stopping=True,
                           )

  decoded += tokenizer.decode(outputs[0], skip_special_tokens=True)+'\n'

# print(decoded)

with open("output.txt", "w") as text_file:
    text_file.write(decoded)

task_prefix = "translate English to German: "
# use different length sentences to test batching
sentences = ["The house is wonderful.", "I like to work in NYC."]

inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)

print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
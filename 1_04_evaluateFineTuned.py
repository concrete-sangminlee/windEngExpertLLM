import transformers
import torch

model_id = "./fine_tuned_llama_A"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "초고층 구조 설계를 위한 풍하중 기준에 대해 알려줘."

full_prompt = f"{PROMPT}\nUser: {instruction}\nAI:"

terminators = [pipeline.tokenizer.eos_token_id]

outputs = pipeline(
    full_prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

generated_text = outputs[0]["generated_text"]
response = generated_text[len(full_prompt):]

if "AI:" in response:
    response = response.split("AI:")[0].strip()

print(response)
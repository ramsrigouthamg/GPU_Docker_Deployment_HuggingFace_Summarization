from fastapi import Request,FastAPI
from pydantic import BaseModel
import uvicorn

from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


app = FastAPI()

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


tokenizer = AutoTokenizer.from_pretrained("./distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("./distilbart-cnn-12-6").to(torch_device)
model = model.to(torch_device)


class SummaryRequest(BaseModel):
    text: str
    min_length: int
    max_length: int

def get_summary(t,tokenizer_summary,model_summary):
  txt = t['text']
  minl = t['min_length'] #75
  maxl = t['max_length'] #150
  inputs = tokenizer_summary([txt], max_length=1024,truncation=True, return_tensors='pt').to(torch_device)
  summary_ids = model_summary.generate(inputs['input_ids'], num_beams=3,num_return_sequences=1,no_repeat_ngram_size=2, min_length = minl,max_length=maxl, early_stopping=True)
  dec = [tokenizer_summary.decode(ids,skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in summary_ids]
  output = dec[0].strip()
  return {'summary':output}


@app.get('/')
async def home():
    return {"message": "Hello World"}

@app.post("/summary")
async def getsummary(user_request_in: SummaryRequest):
    payload = {"text":user_request_in.text,"min_length":user_request_in.min_length,"max_length":user_request_in.max_length}
    summ = get_summary(payload,tokenizer,model)
    summ["Device"]= torch_device
    return summ



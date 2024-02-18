from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Load your model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True)
model.load_state_dict(torch.load('model.pt'))

app = FastAPI()

class Item(BaseModel):
    question: str

@app.post("/predict/")
async def predict(item: Item, file: UploadFile = File(...)):
    # Read image file
    image = await file.read()
    inputs = processor(image.convert('RGB'), text=item.question, return_tensors="pt").to(device, torch.float16)

    # Make prediction
    generated_ids = model.generate(**inputs, max_new_tokens=10)

    # Post processing
    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return {"answer": answer}
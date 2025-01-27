from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Izza-shahzad-13/recipe-generator")
model = AutoModelForSeq2SeqLM.from_pretrained("Izza-shahzad-13/recipe-generator")

# Initialize FastAPI app
app = FastAPI(title="Recipe Generator API")

# Define input schema
class IngredientsInput(BaseModel):
    ingredients: str

# Define a root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the Recipe Generator API!"}

# Endpoint to generate recipes
@app.post("/generate")
def generate_recipe(input: IngredientsInput):
    try:
        # Tokenize the input ingredients
        inputs = tokenizer(f"Ingredients: {input.ingredients}", return_tensors="pt")

        # Generate recipe
        outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=5, early_stopping=True)
        recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"recipe": recipe}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

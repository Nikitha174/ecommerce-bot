import google.generativeai as genai
import json

# Configure your Gemini API key
api_key ="AIzaSyDW8fCzHXtbVDsQ0fk3ku-3X0a0YaTma2Y"

# Configure the API key
genai.configure(api_key=api_key)

# Function to generate intents using Gemini's API
def generate_intents():
    prompt = (
        "Generate a list of intents for an e-commerce chatbot using the Brazilian e-commerce dataset by Olist."
        "Use the product_category_name_translation.csv file to translate product categories into English."
        "Each intent should include a tag, patterns (common user questions in English), and responses (answers from the chatbot in English)."
        "Format the output as valid JSON."
        "Consider common queries about order status, product details, product recommendations, and order tracking."
        "Please ensure all generated content is in English and utilizes the translated category names."
    )

    model = genai.GenerativeModel('gemini-1.5-flash')

    response = model.generate_content(prompt)
    generated_text = response.text

    print("Raw response:",generated_text)  
    mygeneratedtext=generated_text[7:-4]
    print(mygeneratedtext)

    # Attempt to parse the generated text into JSON
    try:
        intents_data = json.loads(mygeneratedtext)
    except Exception as e:
        print("Failed to decode JSON from generated text.")
        print(e)
        return None

    return intents_data

# Generate intents and save to intents.json
intents_data = generate_intents()

if intents_data:
    with open("ecommerce_intents.json", 'w') as json_file:
        json.dump(intents_data, json_file, indent=4)
    print("ecommerce_intents.json has been created successfully!")
else:
    print("No intents data generated.")

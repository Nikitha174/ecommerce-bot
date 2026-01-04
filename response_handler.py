import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import json
import random

# Load ecommerce datasets
try:
    ecommerce_data = pd.read_csv('archive/olist_products_dataset.csv').fillna('')
    product_recommendations = pd.read_excel('product_img.xlsx').fillna('')
    orders = pd.read_csv('archive/olist_orders_dataset.csv').fillna('')
    order_items = pd.read_csv('archive/olist_order_items_dataset.csv').fillna('')
    reviews = pd.read_csv('archive/olist_order_reviews_dataset.csv').fillna('')
    category_translations = pd.read_csv('archive/product_category_name_translation.csv').fillna('')
except FileNotFoundError:
    ecommerce_data = None
    product_recommendations = None
    orders = None
    order_items = None
    reviews = None
    category_translations = None
    print("⚠️ Warning: One or more datasets not found.")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load intents
with open('ecommerce_intents.json', 'r') as f:
    intents = json.load(f)

patterns = []
for intent in intents['intents']:
    patterns.extend(intent['patterns'])

pattern_embeddings = model.encode(patterns, convert_to_tensor=True)


def is_meaningful_input(user_input):
    if not user_input.strip() or not any(char.isalnum() for char in user_input):
        return False
    words = re.findall(r'\b\w+\b', user_input)
    return len(words) > 1


def filter_products_with_embeddings(query, df):
    query_embedding = model.encode(query, convert_to_tensor=True)
    df['combined_attributes'] = df[['COLOR', 'CATEGORY', 'BRAND', 'GENDER', 'PRODUCT TYPE', 'USAGE']].fillna('').agg(' '.join, axis=1)
    df['attribute_embeddings'] = df['combined_attributes'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    df['similarity'] = df['attribute_embeddings'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())

    filtered_df = df[df['similarity'] > 0.3].sort_values(by='similarity', ascending=False)

    keywords = query.split()
    filtered_df = filtered_df[
        filtered_df.apply(
            lambda row: any(re.search(keyword.lower(), row['combined_attributes'].lower()) for keyword in keywords),
            axis=1
        )
    ]
    return filtered_df


def get_response(user_input):
    if orders is not None:
        if "order id" in user_input.lower() or "orderid" in user_input.lower() or "track my order" in user_input.lower() or "order number" in user_input.lower():
            order_id = None
            words = user_input.split()
            for word in words:
                if word.isalnum() and len(word) > 10:  # Assuming order ID is alphanumeric and long
                    order_id = word
                    break

            if order_id:
                order_details = orders[orders['order_id'] == order_id]

                if not order_details.empty:
                    order = order_details.iloc[0]
                    response = f"Order ID: {order_id}\n"
                    response += f"Order Date: {order['order_purchase_timestamp']}\n"
                    response += f"Status: {order['order_status']}\n"

                    # Get order items
                    order_items_details = order_items[order_items['order_id'] == order_id]
                    if not order_items_details.empty:
                        response += "\nOrder Items:\n"
                        for index, item in order_items_details.iterrows():
                            product_id = item['product_id']
                            product = ecommerce_data[ecommerce_data['product_id'] == product_id]

                            if not product.empty:
                                product_row = product.iloc[0]
                                print(product_row)
                                # Ensure the column exists
                                if 'product_category_name' in ecommerce_data.columns:
                                    print("Ecommerce Data:",ecommerce_data)
                                    category = product_row['product_category_name']
                                    print("Looking for category:", category)
                                    #print("Available columns in category_translations:", category_translations.columns.tolist())
                                    #category_translations.columns = category_translations.columns.str.strip().str.lower().str.replace(' ', '_')
                                    category_translations = pd.read_csv('archive/product_category_name_translation.csv', sep='\t').fillna('')
                                    category_translation = category_translations[category_translations['product_category_name'] == category]
                                    if not category_translation.empty:
                                        english_category = category_translation['product_category_name_english'].iloc[0]
                                        response += f"- Category: {english_category}\n Product ID: {product_id}\n Price: {item['price']}\n Freight Value: {item['freight_value']}\n"
                                    else:
                                        response += f"- Category: {category}\n Product ID: {product_id}\n Price: {item['price']}\n Freight Value: {item['freight_value']}\n"
                                else:
                                    response += f"- Product ID: {product_id}\n Price: {item['price']}\n Freight Value: {item['freight_value']}\n"
                            else:
                                response += f"- Product ID: {product_id}\n Price: {item['price']}\n Freight Value: {item['freight_value']}\n"

                    # Get reviews for the order
                    reviews_for_order = reviews[reviews['order_id'] == order_id]
                    if not reviews_for_order.empty:
                        response += "\nOrder Reviews:\n"
                        for index, review in reviews_for_order.iterrows():
                            response += f"- Review Score: {review['review_score']}\n Comment Title: {review['review_comment_title']}\n"

                    return response
                else:
                    return "Sorry, I couldn't find any order with that ID."
            else:
                return "Please provide the order ID."

    # Handle product recommendations
    if product_recommendations is not None:
        if any(word in user_input.lower() for word in ['recommend', 'show', 'product', 'buy', 'suggest']):
            filtered_df = filter_products_with_embeddings(user_input, product_recommendations)
            if not filtered_df.empty:
                response = {'type': 'products', 'products': []}
                for _, product in filtered_df.iterrows():
                    image = product['IMAGE'].replace('\\', '/')
                    if not image.startswith('imgs/'):
                        image = f"imgs/{image}"
                    product_info = {
                        'product_id': product['PRODUCT ID'],
                        'category': product['CATEGORY'],
                        'color': product['COLOR'],
                        'brand': product['BRAND'],
                        'image_path': image
                        }
                    response['products'].append(product_info)
                return response
            else:
                return {'type': 'products', 'products': []}

    # Handle intent-based response
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, pattern_embeddings)
    best_match_index = similarities.argmax()

    best_match_intent = ""
    for intent in intents['intents']:
        if best_match_index < len(intent['patterns']):
            best_match_intent = intent
            break
        best_match_index -= len(intent['patterns'])

    if best_match_intent:
        return random.choice(best_match_intent['responses'])

    return "I'm sorry, I didn't understand that."

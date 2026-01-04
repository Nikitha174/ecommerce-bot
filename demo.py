import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load ecommerce datasets
try:
    ecommerce_data = pd.read_csv('archive/olist_products_dataset.csv')
    ecommerce_data = ecommerce_data.fillna('')
    
    product_recommendations = pd.read_excel('product_img.xlsx')
    product_recommendations = product_recommendations.fillna('')
    
    orders = pd.read_csv('archive/olist_orders_dataset.csv')
    order_items = pd.read_csv('archive/olist_order_items_dataset.csv')
    reviews = pd.read_csv('archive/olist_order_reviews_dataset.csv')
    category_translations = pd.read_csv('archive/product_category_name_translation.csv')
except FileNotFoundError:
    ecommerce_data = None
    product_recommendations = None
    orders = None
    order_items = None
    reviews = None
    category_translations = None
    print("Warning: Some datasets not found.")

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load intents data from intents.json
with open('ecommerce_intents.json', 'r') as f:
    intents = json.load(f)

# Precompute embeddings for all patterns in intents.json
patterns = []
for intent in intents['intents']:
    patterns.extend(intent['patterns'])

# Create embeddings for the patterns
pattern_embeddings = model.encode(patterns, convert_to_tensor=True)

def is_meaningful_input(user_input):
    """Check if the input is a meaningful English phrase."""
    # Check if the input is empty or consists only of symbols
    if not user_input.strip() or not any(char.isalnum() for char in user_input):
        return False
    
    # Check for valid words (using a simple regex to find words)
    words = re.findall(r'\b\w+\b', user_input)  # Find all words in the input
    return len(words) > 1  # Ensure there are at least two words

def filter_products_with_embeddings(query, df):
    """Filter products based on a query using embeddings."""
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Combine relevant columns into a single string for each row
    df['combined_attributes'] = df[['COLOR', 'CATEGORY', 'BRAND', 'GENDER', 'PRODUCT TYPE', 'USAGE']].fillna('').agg(' '.join, axis=1)

    # Generate embeddings for the combined attributes of each product
    df['attribute_embeddings'] = df['combined_attributes'].apply(lambda x: model.encode(x, convert_to_tensor=True))

    # Compute cosine similarity between the query and each product's attributes
    df['similarity'] = df['attribute_embeddings'].apply(lambda x: util.pytorch_cos_sim(query_embedding, x).item())

    # Filter products with a similarity score above a certain threshold (e.g., 0.5)
    filtered_df = df[df['similarity'] > 0.3].sort_values(by='similarity', ascending=False)

    # Additional keyword-based filtering
    keywords = query.split()
    filtered_df = filtered_df[
        filtered_df.apply(
            lambda row: all(
                bool(re.search(keyword.lower(), row['combined_attributes'].lower())) for keyword in keywords
            ), axis=1
        )
    ]

    return filtered_df


def get_response(user_input):
    """Get a response based on user input using NLP techniques."""

    # Check if the user input is meaningful
    if not is_meaningful_input(user_input):
        return "I'm sorry, I didn't understand that."

    # Check for order queries by order ID
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
                                # Ensure the column exists
                                if 'product_category_name' in ecommerce_data.columns:
                                    category = product_row['product_category_name']
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

                    return type(response)
                else:
                    return "Sorry, I couldn't find any order with that ID."
            else:
                return "Please provide the order ID."

    # Check for product queries
    if product_recommendations is not None:
        # Use embeddings to filter products
        filtered_df = filter_products_with_embeddings(user_input, product_recommendations)
        
        if not filtered_df.empty:
            response = f"Here are the products matching '{user_input}':\n"
            for index, product in filtered_df.iterrows():
                print(f"\n- Product ID: {product['PRODUCT ID']}")
                if 'CATEGORY' in product_recommendations.columns:
                    print(f"- Category: {product['CATEGORY']}")
                print(f"- Color: {product['COLOR']}")
                print(f"- Brand: {product['BRAND']}")
                print(f"- Product Type: {product['PRODUCT TYPE']}")
                print(f"- Usage: {product['USAGE']}")
                if 'IMAGE' in product_recommendations.columns:
                    img_path = product['IMAGE']
                    img = mpimg.imread(img_path)
                    plt.imshow(img)
                    plt.show()
            return "Products displayed above."
        else:
            pass
        
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarities between user input and pattern embeddings
    similarities = util.pytorch_cos_sim(user_embedding, pattern_embeddings)

    # Find the index of the highest similarity score
    best_match_index = similarities.argmax()

    # Get the corresponding intent based on the best match index
    best_match_intent = ""
    for intent in intents['intents']:
        if best_match_index < len(intent['patterns']):
            best_match_intent = intent
            break
        best_match_index -= len(intent['patterns'])

    # Return a response from the best matching intent if found
    if best_match_intent:
        return random.choice(best_match_intent['responses'])

    # Default: Use Chatterbot
    else:
        return "I'm sorry, I didn't understand that."

# Example usage with a sample dataset
if __name__ == "__main__":
    # Load the full datasets
    ecommerce_data = pd.read_csv('archive/olist_products_dataset.csv')
    product_recommendations = pd.read_excel('product_img.xlsx')
    
    # Test the filter_products_with_embeddings function
    query = "blue jeans"
    #filtered_df = filter_products_with_embeddings(query, product_recommendations)
    #print(filtered_df)

    # Test the get_response function
    user_input = "women shoe"
    response = get_response(user_input)
    print(response)

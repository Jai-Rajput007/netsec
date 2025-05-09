import pymongo
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
DATABASE_NAME = "NETSEC"  # Replace with your database name
COLLECTION_NAME = "NetworkData"  # Replace with your collection name

client = pymongo.MongoClient(MONGO_DB_URL)
try:
    client.server_info()  # Test connection
    print("Connected to MongoDB")
    collection = client[DATABASE_NAME][COLLECTION_NAME]
    data = list(collection.find())
    print(f"Retrieved {len(data)} documents")
    df = pd.DataFrame(data)
    print("DataFrame shape:", df.shape)
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
finally:
    client.close()
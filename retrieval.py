from database import get_conn
from indexer import model
import numpy
def get_query():
    return input("Enter ur query: ")
def get_embed_from_query(temp,model):
    return model.encode(temp)
def run_retrieval():
    query_string=get_query()
    query_embed=get_embed_from_query(query_string,model)
    with get_conn() as conn:
        cursor=conn.cursor()

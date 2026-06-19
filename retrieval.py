from database import get_conn
from indexer import model
import numpy as np
def get_query():
    return input("Enter ur query: ")
def get_embed_from_query(temp,model):
    return model.encode(temp)
def run_retrieval():
    query_string=get_query()
    query_embed=get_embed_from_query(query_string,model)
    with get_conn() as conn:
        cursor=conn.cursor()
        query="SELECT docs.file_path,chunks.chunk_text,chunks.vector FROM chunks JOIN docs ON chunks.doc_id=docs.id"
        cursor.execute(query)
        rows=cursor.fetchall()
        max_sim=-1.0
        best_string=None
        best_path=None
        for row in rows:
            vec=np.frombuffer(row[2],dtype="float32")
            similarity=np.dot(vec,query_embed)
            if(similarity>max_sim):
                max_sim=similarity
                best_string=row[1]
                best_path=row[0]
    return max_sim,best_string,best_path
        #print("\n--- Best Local Match ---")
        #print(f"File Source: {best_path}")
        #print(f"Similarity Score: {max_sim:.4f}")
        #print(f"Content: {best_string}\n")

if __name__=="__main__":
    run_retrieval()



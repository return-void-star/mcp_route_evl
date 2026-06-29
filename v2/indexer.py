import os
import hashlib
import re
import numpy as np
from database import get_conn
accepted_file_types=(".md",".txt")
def split_sent(string:str)->list:
    chunks = re.split(r'\n\s*\n|(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', string)
    return [c.strip() for c in chunks if c.strip()]
def sem_chunking(model,temp,threshold):
    temp=split_sent(temp)
    temp_embed_list=model.encode(temp)
    similarity_scores=np.sum(temp_embed_list[:-1]*temp_embed_list[1:],axis=1)
    split_arr=[i+1 for i in range(len(similarity_scores)) if similarity_scores[i]<threshold]
    temp_embed_list=np.split(temp_embed_list,split_arr)
    temp_chunked=np.split(temp,split_arr)
    chun_vec=[]
    for x in temp_embed_list:
        mean_vec=np.mean(x,axis=0)
        norm=np.linalg.norm(mean_vec)
        if(norm>0):
            mean_vec=mean_vec/norm
        chun_vec.append(mean_vec)
    return temp_chunked,chun_vec

def crawler(path):
    file_paths=[]
    for root,dir, files in os.walk(path):
        for file in files:
            if(file[0]==".") or not file.endswith(accepted_file_types):
                continue
            file_path=os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths
'''
def run_indexer(model,path):
    if os.path.isdir(path):
        file_paths=crawler(path)
        file_paths=list(file_paths)
        chunk_size=65536 #64 kb
        with get_conn() as temp_conn:
            for p in file_paths:
                hasher = hashlib.sha256()
                text="" #now whole file text is going to be here, not good for optimzation, need to fix this, later
                with open(p,"r",encoding="utf-8",errors="ignore") as temp:
                    while True:
                        chunk=temp.read(chunk_size)
                        text+=chunk
                        if not chunk:
                            break
                        hasher.update(chunk.encode("utf-8"))
                final_hash=hasher.hexdigest()
                cursor=temp_conn.cursor()
                cursor.execute("SELECT file_hash FROM docs WHERE file_path=?",(p,))
                row=cursor.fetchone()
                if(row is not None): #this file_path is in db
                    data_hash=row[0]
                    if(data_hash==final_hash): #no change in file
                        continue
                    else: #file changed
                        cursor.execute("DELETE FROM docs WHERE file_path =?",(p,))
                p_parent_path=os.path.basename(os.path.dirname(p))
                p_extension=os.path.splitext(p)[1]
                query="INSERT INTO docs(file_path,parent_folder,file_extension,file_hash) VALUES(?,?,?,?)"
                cursor.execute(query,(p,p_parent_path,p_extension,final_hash))
                chun_thres=0.6 # chunking threshold; can we make it dynamic??
                text_chunked,chunk_embed=sem_chunking(model,text,chun_thres) #semantic chunking
                p_docs_id=cursor.lastrowid
                query="INSERT INTO chunks(doc_id,chunk_text,chunk_index,vector) VALUES(?,?,?,?)"
                for i,vec in enumerate(chunk_embed):
                    cursor.execute(query,(p_docs_id," ".join(text_chunked[i]),i,vec.astype("float32").tobytes()))
            temp_conn.commit()
    else:
        print(f"{path} is not a valid directory")
        return
'''
def index_after_modification(file_path,model):
    chunk_size = 65536  # 64 kb
    text=""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as temp:
        while True:
            chunk = temp.read(chunk_size)
            text += chunk
            if not chunk:
                break
    with get_conn() as temp_conn:
        cursor=temp_conn.cursor()
        cursor.execute("DELETE FROM docs WHERE file_path =?", (file_path,))
        parent_path = os.path.basename(os.path.dirname(file_path))
        extension = os.path.splitext(file_path)[1]
        query = "INSERT INTO docs(file_path,parent_folder,file_extension) VALUES(?,?,?)"
        cursor.execute(query, (file_path, parent_path, extension))
        chun_thres = 0.6  # chunking threshold; can we make it dynamic??
        text_chunked, chunk_embed = sem_chunking(model, text, chun_thres)  # semantic chunking
        p_docs_id = cursor.lastrowid
        query = "INSERT INTO chunks(doc_id,chunk_text,chunk_index,vector) VALUES(?,?,?,?)"
        for i, vec in enumerate(chunk_embed):
            cursor.execute(query, (p_docs_id, " ".join(text_chunked[i]), i, vec.astype("float32").tobytes()))
        temp_conn.commit()

def delete_index(file_path):
    with get_conn() as temp_conn:
        cursor=temp_conn.cursor()
        cursor.execute("DELETE FROM docs WHERE file_path =?", (file_path,))
        temp_conn.commit()

if __name__=="__main__":
    from sentence_transformers import SentenceTransformer
    mdl = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    '''run_indexer(model,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data"))'''


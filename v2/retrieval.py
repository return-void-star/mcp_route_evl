from database import get_conn
import numpy as np
import re
#print("\n--- Best Local Match ---")
#print(f"File Source: {best_path}")
#print(f"Similarity Score: {max_sim:.4f}")
#print(f"Content: {best_string}\n")

'''def search_locally(query_embed,q_text): #hybrid: semantic and keyword search
    fts_query= " OR ".join(re.findall(r"\w+", q_text)) if re.findall(r"\w+", q_text) else ""
    with get_conn() as conn:
        cursor = conn.cursor()
        query = "SELECT chunks.id,docs.file_path,chunks.char_start,chunks.char_end,chunks.vector FROM chunks JOIN docs ON chunks.doc_id=docs.id"
        cursor.execute(query)
        rows = cursor.fetchall()
        vec_sims=[]
        for row in rows:
            chunk_id,file_path,char_start,char_end,vec_blob=row
            vec=np.frombuffer(vec_blob,dtype="float32")
            similarity=float(np.dot(vec,query_embed))
            vec_sims.append((similarity,chunk_id,file_path,char_start,char_end,vec))
        vec_sims.sort(key=lambda x:x[0],reverse=True)
        best_vecs=vec_sims[:20]
        vec_ranks={}
        for rank, item in enumerate(best_vecs,start=1):
            similarity,chunk_id,file_path,char_start,char_end,vec=item
            vec_ranks[chunk_id]=(rank,similarity,file_path,char_start,char_end,vec)
        fts_ranks={}
        if fts_query:
            try:
                cursor.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 20",(fts_query,))
                for rank, row in enumerate(cursor.fetchall(),start=1):
                    fts_ranks[row[0]]=rank
            except Exception as e:
                print(f"FTS5 Query failed: {e}")
        all_chunk_ids=set(vec_ranks.keys()).union(set(fts_ranks.keys()))
        rrf_scores=[]
        k=60 #rrf constamt
        for chunk_id in all_chunk_ids:
            r_vec=vec_ranks[chunk_id][0] if chunk_id in vec_ranks else float("inf")
            r_key=fts_ranks[chunk_id] if chunk_id in fts_ranks else float("inf")
            score=(1.0/(k+r_vec))+(1.0/(k+r_key))
            rrf_scores.append((score,chunk_id))

        if not rrf_scores:
            return -1, None, None

        rrf_scores.sort(key=lambda x: x[0],reverse=True)
        winner_chunk_id=rrf_scores[0][1]
        if winner_chunk_id in vec_ranks:
            _, similarity,best_path,char_start,char_end,_=vec_ranks[winner_chunk_id]
        else:
            cursor.execute("SELECT docs.file_path, chunks.char_start, chunks.char_end,chunks.vector FROM chunks JOIN docs ON chunks.doc_id=docs.id WHERE chunks.id=? ",(winner_chunk_id,))
            best_path,char_start,char_end,vec_blob=cursor.fetchone()
            vec=np.frombuffer(vec_blob,dtype="float32")
            similarity=float(np.dot(vec,query_embed))
        try:
            with open(best_path,"r",encoding="utf-8",errors="ignore") as temp:
                text=temp.read()
                best_string=text[char_start:char_end]
        except Exception as e:
            best_string="[Error reading file content]"

    return similarity,best_string,best_path'''

def search_locally(query_embed,q_text): #hybrid: semantic and keyword search; falls back to semantic if keyword not found
    words=re.findall(r"\w+", q_text)
    with get_conn() as conn:
        cursor=conn.cursor()
        if words:
            phrase_query=f'"{" ".join(words)}"'
            try:
                cursor.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1", (phrase_query,))
                row=cursor.fetchone()
                if row:
                    winner_chunk_id=row[0]
                    cursor.execute("SELECT docs.file_path, chunks.char_start, chunks.char_end FROM chunks JOIN docs ON chunks.doc_id=docs.id WHERE chunks.id=?",(winner_chunk_id,))
                    row=cursor.fetchone()
                    if row:
                        best_path,char_first,char_end=row
                    try:
                        with open(best_path,"r",encoding="utf-8",errors="ignore") as f:
                            text=f.read()
                            best_string=text[char_first:char_end]
                    except Exception as e:
                        best_string="[Error reading file content]"
                    return 1.0, best_string,best_path
            except Exception as e:
                print(f"Exact phrase bypass query failed: {e}")

        fts_query = " OR ".join(re.findall(r"\w+", q_text)) if re.findall(r"\w+", q_text) else ""
        query = "SELECT chunks.id,docs.file_path,chunks.char_start,chunks.char_end,chunks.vector FROM chunks JOIN docs ON chunks.doc_id=docs.id"
        cursor.execute(query)
        rows = cursor.fetchall()
        vec_sims = []
        for row in rows:
            chunk_id, file_path, char_start, char_end, vec_blob = row
            vec = np.frombuffer(vec_blob, dtype="float32")
            similarity = float(np.dot(vec, query_embed))
            vec_sims.append((similarity, chunk_id, file_path, char_start, char_end, vec))
        vec_sims.sort(key=lambda x: x[0], reverse=True)
        best_vecs = vec_sims[:20]
        vec_ranks = {}
        for rank, item in enumerate(best_vecs, start=1):
            similarity, chunk_id, file_path, char_start, char_end, vec = item
            vec_ranks[chunk_id] = (rank, similarity, file_path, char_start, char_end, vec)
        fts_ranks = {}
        if fts_query:
            try:
                cursor.execute(
                    "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT 20",
                    (fts_query,))
                for rank, row in enumerate(cursor.fetchall(), start=1):
                    fts_ranks[row[0]] = rank
            except Exception as e:
                print(f"FTS5 Query failed: {e}")
        all_chunk_ids = set(vec_ranks.keys()).union(set(fts_ranks.keys()))
        rrf_scores = []
        k = 60  # rrf constamt
        for chunk_id in all_chunk_ids:
            r_vec = vec_ranks[chunk_id][0] if chunk_id in vec_ranks else float("inf")
            r_key = fts_ranks[chunk_id] if chunk_id in fts_ranks else float("inf")
            score = (1.0 / (k + r_vec)) + (1.0 / (k + r_key))
            rrf_scores.append((score, chunk_id))

        if not rrf_scores:
            return -1, None, None

        rrf_scores.sort(key=lambda x: x[0], reverse=True)
        winner_chunk_id = rrf_scores[0][1]
        if winner_chunk_id in vec_ranks:
            _, similarity, best_path, char_start, char_end, _ = vec_ranks[winner_chunk_id]
        else:
            cursor.execute(
                "SELECT docs.file_path, chunks.char_start, chunks.char_end,chunks.vector FROM chunks JOIN docs ON chunks.doc_id=docs.id WHERE chunks.id=? ",
                (winner_chunk_id,))
            best_path, char_start, char_end, vec_blob = cursor.fetchone()
            vec = np.frombuffer(vec_blob, dtype="float32")
            similarity = float(np.dot(vec, query_embed))
        try:
            with open(best_path, "r", encoding="utf-8", errors="ignore") as temp:
                text = temp.read()
                best_string = text[char_start:char_end]
        except Exception as e:
            best_string = "[Error reading file content]"

    return similarity, best_string, best_path









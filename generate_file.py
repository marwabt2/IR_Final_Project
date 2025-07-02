import json

# تحميل qrels
qrels_path = r'C:\Users\USER\.ir_datasets\antique\test\qrels'
queries_path =r'C:\Users\USER\.ir_datasets\antique\test\queries.txt'
output_path = r'Answers.jsonl'

# نحمّل qrels كقاموس: {query_id: [doc_id1, doc_id2, ...]}
qrels = {}
with open(qrels_path, 'r', encoding='utf-8') as f:
    for line in f:
        qid, _, doc_id, relevance = line.strip().split()
        if int(relevance) > 0:  # نحتفظ فقط بالوثائق المتعلقة
            qrels.setdefault(qid, []).append(doc_id)

# نحمّل الاستعلامات ونبدأ بترقيمهم من 0
qid_map = {}
queries = []
with open(queries_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        qid, query_text = line.strip().split('\t', 1)
        qid_map[qid] = i  # نحول الـ original_qid إلى رقم جديد
        queries.append((qid, query_text))

# نكتب الملف الناتج
with open(output_path, 'w', encoding='utf-8') as out:
    for original_qid, query_text in queries:
        new_qid = qid_map[original_qid]
        answer_pids = qrels.get(original_qid, [])
        item = {
            "qid": new_qid,
            "query": query_text,
            "url": f"https://fakeurl.org/query/{original_qid}",
            "answer_pids": answer_pids
        }
        out.write(json.dumps(item) + '\n')

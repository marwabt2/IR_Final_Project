import json
from pymongo import MongoClient

def load_queries_to_mongo(file_path: str, collection_name: str):
    """
    تحميل استعلامات من ملف JSONL إلى MongoDB
    وتخزينها في تجميعة محددة، مع الاحتفاظ فقط بـ qid و query.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            obj = json.loads(line)
            data.append({
                "qid": obj["qid"],
                "query": obj["query"]
            })

    client = MongoClient("mongodb://localhost:27017/")
    db = client["query_db"]
    collection = db[collection_name]

    collection.delete_many({})
    collection.insert_many(data)

    print(f"✅ تم إدخال {len(data)} استعلام إلى MongoDB في التجميعة '{collection_name}' بنجاح.")

# الاستدعاء من main
if __name__ == "__main__":
    # path = r'C:\Users\USER\.ir_datasets\lotte\lotte_extracted\lotte\lifestyle\dev\qas.search.jsonl'
    # collection = "queries_lifestyle" 
    path = r'C:\Users\USER\.ir_datasets\antique\test\Answers.jsonl'
    collection = "queries_antique" 
    load_queries_to_mongo(path, collection)

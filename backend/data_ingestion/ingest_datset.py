# import sys
# import os
# sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

# import ir_datasets
# from backend.database.connection import get_mongo_connection
# def ingest_dataset(dataset_path: str):
#     print(f"✅ Start Loading '{dataset_path}'")
#     dataset = ir_datasets.load(dataset_path)

#     db = get_mongo_connection()
#     collection_name = dataset_path.replace("/", "_")
#     collection = db[collection_name]

#     count = 0
#     for doc in dataset.docs_iter():
#         doc_dict = {
#             "doc_id": doc.doc_id,
#             "text": doc.text
#         }
#         collection.insert_one(doc_dict)
#         count += 1

#     print(f"✅ Dataset '{dataset_path}' has been ingested into collection '{collection_name}'")
#     print(f"✅ {count} doc has been ingested")
# if __name__ == "__main__":
#     ingest_dataset("antique/train")

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))

import ir_datasets
from backend.database.connection import get_mongo_connection
from backend.logger_config import logger

def ingest_dataset(dataset_path: str, batch_size: int = 10000):
    try:
        logger.info(f"✅ Start loading dataset: '{dataset_path}'")
        dataset = ir_datasets.load(dataset_path)

        db = get_mongo_connection()
        collection_name = dataset_path.replace("/", "_")
        collection = db[collection_name]

        # ❗احذري: هذا يحذف البيانات القديمة من المجموعة إذا كانت موجودة
        # collection.drop()

        batch = []
        count = 0

        for doc in dataset.docs_iter():
            doc_dict = {
                "doc_id": doc.doc_id,
                "text": doc.text
            }
            batch.append(doc_dict)

            if len(batch) >= batch_size:
                collection.insert_many(batch)
                count += len(batch)
                logger.info(f"🔄 Inserted {count} documents so far...")
                batch = []

        # إدخال ما تبقى من الوثائق
        if batch:
            collection.insert_many(batch)
            count += len(batch)

        logger.info(f"✅ Finished ingestion for dataset '{dataset_path}'")
        logger.info(f"✅ Total documents ingested: {count}")

    except Exception as e:
        logger.error(f"🔥 Error during ingestion of '{dataset_path}': {str(e)}")

if __name__ == "__main__":
    #ingest_dataset("antique/train")
    ingest_dataset("lotte/lifestyle/dev/forum")



import pyterrier as pt
if not pt.started():
    pt.init()
import pandas as pd

def process_ds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns processed dataset data. We filter the dataframes so we only use queries
    with qrels defined, then join the TREC 2019 and 2020 test data together.
    Returns the queries and qrels.
    """
    trec_ds = pt.get_dataset("trec-deep-learning-passages")

    qrels_2019 = trec_ds.get_qrels("test-2019")
    qids_2019 = [item[0] for item in qrels_2019.groupby("qid")["qid"]]
    topic_2019 = trec_ds.get_topics("test-2019")
    topic_2019[topic_2019["qid"].isin(qids_2019)]
    
    qrels_2020 = trec_ds.get_qrels("test-2020")
    qids_2020 = [item[0] for item in qrels_2020.groupby("qid")["qid"]]
    topic_2020 = trec_ds.get_topics("test-2020")
    topic_2020[topic_2020["qid"].isin(qids_2020)]

    topic = pd.concat([topic_2019, topic_2020])
    qrels = pd.concat([qrels_2019, qrels_2020])

    return topic, qrels
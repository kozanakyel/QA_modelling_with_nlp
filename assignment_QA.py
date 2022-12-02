import nltk
import spacy
nlp = spacy.load('en_core_web_md')
import pandas as pd
import re
import warnings
from nltk.corpus import stopwords
from textblob import TextBlob
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering, pipeline
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from xgboost import XGBClassifier
import numpy as np

"""
@author: Ugur AKYEL
@no: 20190808020
@subject: Project assignment for QA modelling in NLP
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)


def cleaning_data(df: pd.DataFrame()):
    df_c = df.copy()
    df_c.dropna(inplace=True)
    blanks = []

    for i, ans_ind, answers in df_c.itertuples():  
        if type(answers)==str:            
            if answers.isspace():         
                blanks.append(i)    

    df_c.drop(blanks, inplace=True)
    df_c['answer'] = df_c['answer'].str.lower()
    df_c = df_c.apply(lambda x: x.astype(str).str.lower()).drop_duplicates(keep='first')
    df_c = df_c.reset_index(drop=True)
    return df_c

def add_sent_numbers(df_temp: pd.DataFrame()):
    df_temp['sent_numbers'] = 0
    for i, ans_ind, answers, sent_no in df_temp.itertuples():  
        doc_temp = nlp(answers)
        k = 0
        for j in doc_temp.sents:
            k += 1
        df_temp.at[i, 'sent_numbers'] = k

def uncommon_words(A, B):
    count = {}
    for word in A.split():
        count[word] = count.get(word, 0) + 1
    for word in B.split():
        count[word] = count.get(word, 0) + 1
    return [word for word in count if count[word] == 1]

def auto_correct_spelling_words(df_t: pd.DataFrame()):
    spelling_words = []
    for i in df_t.index:
        cleaning_answer = re.findall("[a-zA-Z,.]+",df_t.answer[i])
        updated_ans = (" ".join(cleaning_answer))
        temp_ans = str(TextBlob(updated_ans).correct())
        spelling_words += uncommon_words(temp_ans, df_t.answer[i])
        df_t.answer[i] = temp_ans

def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words

def get_similarities_correct_answers(answer: str, df_t: pd.DataFrame()):
    df_t['similarity_to_true'] = np.nan
    for i in df_t.index:
        res = nlp(answer).similarity(nlp(df_t.answer[i]))
        df_t.at[i, 'similarity_to_true'] = res

def get_score_with_BERT_SQUAD(df_t: pd.DataFrame()) -> list:

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

    score_list = []
    for index, ans_ind, ans, *sent in df_t.itertuples():
        question, text = "How does the abstraction helps engineering ?", df_t.answer[index]

        question_answerer = pipeline("question-answering", model = model, tokenizer= tokenizer)

        result = question_answerer(question=question, context = text)
        score_list.append(
            result
        )
    return score_list

def add_score_and_status_binary(scores: list, df_t: pd.DataFrame()):
    df_test = pd.DataFrame(scores)
    df_t["score_label"] = df_test['score']
    mean_score = df_t['score_label'].mean()
    df_t['status'] = (df_t['score_label'] > mean_score).astype(int)

def get_similarities_correct_answers(answer: str, df_t: pd.DataFrame()):
    df_t['similarity_to_true'] = np.nan
    for i in df_t.index:
        res = nlp(answer).similarity(nlp(df_t.answer[i]))
        df_t.at[i, 'similarity_to_true'] = res

def extract_answer_from_complete_text(df_t: pd.DataFrame()):
    answer_text = ' '.join(df_t.answer.to_list())
    qa_model = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
    question = "How does the abstraction helps engineering ?"
    context = answer_text
    result = qa_model(question=question, context=context)
    correct_answer_index = int(str(df_t.loc[df_t['answer'].str.contains(result['answer']), 'answer'])[:3])
    correct_answer_sentences = df_t.loc[correct_answer_index, 'answer']
    return correct_answer_sentences

def create_pipeline_tfid_and_model(model, df_t: pd.DataFrame(), model_name: str):
    X = df_t['answer']
    y = df_t['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', model),
                    ])
    text_clf.fit(X_train, y_train) 
    predictions = text_clf.predict(X_test)
    print(f'\nCondusion Matrix for {model_name}')
    print(metrics.confusion_matrix(y_test,predictions))
    print(f'\nClassification report for {model_name}')
    print(metrics.classification_report(y_test,predictions))
    print(f'\nAccuracy Score for {model_name}')
    print(metrics.accuracy_score(y_test,predictions))

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = tf.cast(tf.tile(tf.expand_dims(attention_mask, -1), [1, 1, token_embeddings.shape[-1]]), tf.float32)
    return tf.math.reduce_sum(token_embeddings * input_mask_expanded, 1) / tf.math.maximum(tf.math.reduce_sum(input_mask_expanded, 1), 1e-9)

def encode(texts, tokenizer, model):
    
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')
    model_output = model(**encoded_input, return_dict=True)

    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    embeddings = tf.math.l2_normalize(embeddings, axis=1)

    return embeddings

def get_multi_qa_score(df_t: pd.DataFrame(), query: str):
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    model = TFAutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
    df_t['get_multiqa_score'] = 0
    for index, ans_ind, ans, sent, *scr in df_t.itertuples():
        query_emb = encode(query, tokenizer, model)
        doc_emb = encode([df_t.answer[index]], tokenizer, model)


        scores = (query_emb @ tf.transpose(doc_emb))[0].numpy().tolist()
        doc_score_pairs = list(zip([df_t.answer[index]], scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        for doc, score in doc_score_pairs:
            df_t.loc[index, 'get_multiqa_score'] = score
            #print(score, doc)

def get_result_with_mean(df_t: pd.DataFrame()):
    df_t['get_mean_all_results'] = (df_t['similarity_to_true'] + df_t['score_label'] + df_t['get_multiqa_score'])/3

if __name__ == '__main__':
    df = pd.read_csv('/mnt/c/Users/kozan/Desktop/NLP/QA_modelling_with_nlp/QAs.txt',usecols=[0,1], 
            names=['ans_index', 'answer'], header=None, sep='\t')
    df = cleaning_data(df)
    add_sent_numbers(df)
    auto_correct_spelling_words(df)

    print(f'GIVE AN CORRECT ANSWER AND TYPE **ENTER** FOR INPUT:\n')
    query = input()

    question = "How does the abstraction helps engineering ?"
    get_multi_qa_score(df, question)

    score_list = get_score_with_BERT_SQUAD(df)
    add_score_and_status_binary(score_list, df)
    
    correct_answer_sentences = extract_answer_from_complete_text(df)
    print(f'\n\n#############Correct Answer From compound Answer Text is ########### \n\"{correct_answer_sentences}\"')
    get_similarities_correct_answers(query, df)

    create_pipeline_tfid_and_model(LinearSVC(), df, 'LinearSVC')
    create_pipeline_tfid_and_model(XGBClassifier(), df, 'Xgboost')

    get_result_with_mean(df)

    df.to_csv('/mnt/c/Users/kozan/Desktop/NLP/project_assignment/result_table.csv')
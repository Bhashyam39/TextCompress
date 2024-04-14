import numpy as np
import pandas as pd
import re
import os
import cloudpickle
from transformers import (DebertaTokenizerFast, 
                          TFAutoModelForTokenClassification,
                          BartTokenizerFast, 
                          TFAutoModelForSeq2SeqLM)
import tensorflow as tf
import spacy
import streamlit as st


class NERLabelEncoder:
    '''
    Label Encoder to encode and decode the entity labels
    '''
    def __init__(self):
        self.label_mapping = {'O': 0, 
                             'B-geo': 1, 
                             'I-geo': 2, 
                             'B-gpe': 3, 
                             'I-gpe': 4, 
                             'B-per': 5,
                             'I-per': 6,
                             'B-org': 7,
                             'I-org': 8,
                             'B-tim': 9,
                             'I-tim': 10,
                             'B-art': 11, 
                             'I-art': 12,
                             'B-nat': 13,
                             'I-nat': 14,
                             'B-eve': 15,
                             'I-eve': 16,
                             '[CLS]': -100,
                             '[SEP]': -100}
        
        self.inverse_label_mapping = {}
    
    def fit(self):
        self.inverse_label_mapping = {value: key for key, value in self.label_mapping.items()}
        return self
        
    def transform(self, x: pd.Series):
        x = x.map(self.label_mapping)
        return x
    
    def inverse_transform(self, x: pd.Series):
        x = x.map(self.inverse_label_mapping)
        return x


############ NER MODEL & VARS INITIALIZATION START ####################
NER_CHECKPOINT = "microsoft/deberta-base"
NER_N_TOKENS = 50
NER_N_LABELS = 18
NER_COLOR_MAP = {'GEO': '#DFFF00', 'GPE': '#FFBF00', 'PER': '#9FE2BF', 
                 'ORG': '#40E0D0', 'TIM': '#CCCCFF', 'ART': '#FFC0CB', 'NAT': '#FFE4B5', 'EVE': '#DCDCDC'}

@st.cache_resource
def load_ner_models():
    ner_model = TFAutoModelForTokenClassification.from_pretrained(NER_CHECKPOINT, num_labels=NER_N_LABELS, attention_probs_dropout_prob=0.4, hidden_dropout_prob=0.4)
    ner_model.load_weights(os.path.join("models", "general_ner_deberta_weights.h5"), by_name=True)
    ner_label_encoder = NERLabelEncoder()
    ner_label_encoder.fit()
    ner_tokenizer = DebertaTokenizerFast.from_pretrained(NER_CHECKPOINT, add_prefix_space=True)
    nlp = spacy.load(os.path.join('.', 'en_core_web_sm-3.6.0'))
    print('Loaded NER models')
    return ner_model, ner_label_encoder, ner_tokenizer, nlp

ner_model, ner_label_encoder, ner_tokenizer, nlp = load_ner_models()

############ NER MODEL & VARS INITIALIZATION END ####################

############ NER LOGIC START ####################
def softmax(x):
    return tf.exp(x) / tf.math.reduce_sum(tf.exp(x))

def ner_process_output(res):
    '''
    Function to concatenate sub-word tokens, labels and 
    compute mean prediction probability of tokens
    '''
    d = {}
    result = []
    pred_prob = []
    res.append(['-', 'B-b', 0])
    for n, i in enumerate(res):
        try:
            split = i[1].split('-')
            token = i[0]
            token_prob = i[2]
            prefix, suffix = split
            if prefix == 'B':
                if len(d) != 0:
                    result.append([(re.sub(r"[^\x00-\x7F]+", '', token.replace("Ġ", " ").strip()), label, np.mean(pred_prob))
                                   for label, token in d.items()][0])
                d = {}
                pred_prob = []
                pred_prob.append(token_prob)
                d[suffix] = token

            else:
                d[suffix] = d[suffix] + token
                pred_prob.append(token_prob)
        except:
            continue
            
    return result


def ner_inference(txt):
    '''
    Function that returns model prediction and prediction probabitliy
    '''
    test_data = [txt]
    # tokenizer = DebertaTokenizerFast.from_pretrained(NER_CHECKPOINT, add_prefix_space=True)
    tokens = ner_tokenizer.tokenize(txt)
    tokenized_data = ner_tokenizer(test_data, is_split_into_words=True, max_length=NER_N_TOKENS, 
                               truncation=True, padding="max_length")

    token_idx_to_consider = tokenized_data.word_ids()
    token_idx_to_consider = [i for i in range(len(token_idx_to_consider)) if token_idx_to_consider[i] is not None] 

    input_ = [tokenized_data['input_ids'], tokenized_data['attention_mask']]
    pred_logits = ner_model.predict(input_, verbose=0).logits[0]

    pred_prob = tf.map_fn(softmax, pred_logits)

    pred_idx = tf.argmax(pred_prob, axis=-1).numpy()
    pred_idx = pred_idx[token_idx_to_consider]

    pred_prob = tf.math.reduce_max(pred_prob, axis=-1).numpy()
    pred_prob = np.round(pred_prob[token_idx_to_consider], 3)
    pred_labels = ner_label_encoder.inverse_transform(pd.Series(pred_idx))

    result = [[token, label, prob] for token, label, 
              prob in zip(tokens, pred_labels, pred_prob) if label.find('-') >= 0]
    
    output = ner_process_output(result)
    return output


def ner_inference_long_text(txt):
    entities = []
    doc = nlp(txt)
    for sent in doc.sents:
        entities.extend(ner_inference(sent.text))
    return entities


def get_ner_text(article_txt, ner_result):
    res_txt = ''
    start = 0
    prev_start = 0
    for i in ner_result:
        try:
            span = next(re.finditer(fr'{i[0]}', article_txt)).span()
            start = span[0]
            end = span[1]
            res_txt += article_txt[prev_start:start]
            repl_str = f'''<span style="background-color:{NER_COLOR_MAP[i[1]]}; border-radius: 3px">{article_txt[start:end].strip()}
            <span style="font-size:10px; font-weight:bold; display:inline-block; vertical-align: middle;">
            {i[1]} ({str(np.round(i[2], 3))})</span></span>'''
            res_txt += article_txt[start:end].replace(article_txt[start:end], repl_str)
            prev_start = 0
            article_txt = article_txt[end:]
        except:
            continue
    res_txt += article_txt
    return res_txt

############ NER LOGIC END ####################

############ SUMMARIZATION MODEL & VARS INITIALIZATION START ####################
SUMM_CHECKPOINT = "facebook/bart-base"
SUMM_INPUT_N_TOKENS = 400
SUMM_TARGET_N_TOKENS = 100

@st.cache_resource
def load_summarizer_models():
    summ_tokenizer = BartTokenizerFast.from_pretrained(SUMM_CHECKPOINT)
    summ_model = TFAutoModelForSeq2SeqLM.from_pretrained(SUMM_CHECKPOINT)
    summ_model.load_weights(os.path.join("models", "bart_en_summarizer.h5"), by_name=True)
    print('Loaded summarizer models')
    return summ_tokenizer, summ_model

summ_tokenizer, summ_model = load_summarizer_models()

def summ_preprocess(txt):
    txt = re.sub(r'^By \. [\w\s]+ \. ', ' ', txt) # By . Ellie Zolfagharifard . 
    txt = re.sub(r'\d{1,2}\:\d\d [a-zA-Z]{3}', ' ', txt) # 10:30 EST
    txt = re.sub(r'\d{1,2} [a-zA-Z]+ \d{4}', ' ', txt) # 10 November 1990
    txt = txt.replace('PUBLISHED:', ' ')
    txt = txt.replace('UPDATED', ' ')
    txt = re.sub(r' [\,\.\:\'\;\|] ', ' ', txt) # remove puncts with spaces before and after
    txt = txt.replace(' : ', ' ')
    txt = txt.replace('(CNN)', ' ')
    txt = txt.replace('--', ' ')
    txt = re.sub(r'^\s*[\,\.\:\'\;\|]', ' ', txt) # remove puncts at beginning of sent
    txt = re.sub(r' [\,\.\:\'\;\|] ', ' ', txt) # remove puncts with spaces before and after
    txt = re.sub(r'\n+',' ', txt)
    txt = " ".join(txt.split())
    return txt

def summ_inference_tokenize(input_: list, n_tokens: int):
    tokenized_data = summ_tokenizer(text=input_, max_length=SUMM_TARGET_N_TOKENS, truncation=True, padding="max_length", return_tensors="tf")
    return summ_tokenizer, tokenized_data    

def summ_inference(txt: str):
    txt = summ_preprocess(txt)
    test_data = [txt]
    inference_tokenizer, tokenized_data = summ_inference_tokenize(input_=test_data, n_tokens=SUMM_INPUT_N_TOKENS)
    pred = summ_model.generate(**tokenized_data, max_new_tokens=SUMM_TARGET_N_TOKENS)
    result = inference_tokenizer.decode(pred[0])
    result = re.sub("<.*?>", "", result).strip()
    return result
############ SUMMARIZATION MODEL & VARS INITIALIZATION END ####################

############## ENTRY POINT START #######################
def main():
    st.markdown('''<h3>Text Summarizer</h3>
    #<p><a href="https://huggingface.co/spaces/Sravan1214/news-summarizer-ner/blob/main/README.md" target="_blank">README</a></p>''', unsafe_allow_html=True)
    article_txt = st.text_area("Paste the text (the longer, the better):", "", height=200)
    article_txt = re.sub(r'\n+',' ', article_txt)
    if st.button("Submit"):
        ner_result = [[ent, label.upper(), np.round(prob, 3)] 
                                  for ent, label, prob in ner_inference_long_text(article_txt)]
        
        ner_df = pd.DataFrame(ner_result, columns=['entity', 'label', 'confidence'])
        summ_result = summ_inference(article_txt)        
        
        ner_txt = get_ner_text(article_txt, ner_result).replace('$', '\$')

        st.markdown(f"<h4>SUMMARY:</h4>{summ_result}", unsafe_allow_html=True)

############## ENTRY POINT END #######################

if __name__ == "__main__":
    main()
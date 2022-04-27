import streamlit as st
from annotated_text import annotation
import streamlit.components.v1 as components
import time
import pandas as pd
import numpy as np
import requests
import base64
import webbrowser

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from wordcloud import WordCloud
import matplotlib.pyplot as plt

DEFAULT_NUMBER_OF_ANSWERS = 3
DEFAULT_DOCS_FROM_RETRIEVER = 6

wc_params = dict(
    # background_color = "black",
    background_color = "white",
    width = 1000, height = 1000, margin = 5,
    max_words = 200, min_font_size = 1, relative_scaling = 0.75
)
wc = WordCloud(**wc_params)
ps = PorterStemmer()

def norm_token(token):
    token = token.lower()
    return ps.stem(token)
def get_found_token(query, text):
    token_text_list = word_tokenize(text)
    token_query_list = word_tokenize(query)

    norm_token_text_list = [norm_token(each) for each in token_text_list]
    norm_query_text_list = [norm_token(each) for each in token_query_list]

    text_dict = {}
    for i, each in enumerate(token_text_list):
        text_dict[i] = each

    found_token_list = []
    for each in norm_query_text_list:
        if each in norm_token_text_list:
            found_token_list.append(each)
    found_token_list = [each for each in found_token_list if each not in stopwords.words('english')]
    found_token_list = list(set(found_token_list))
    found_token_index_list = [i for i, each in enumerate(norm_token_text_list) if each in found_token_list]

    found_token_index_res_list = [text_dict[each] for each in found_token_index_list]
    found_token_index_res_list = list(set(found_token_index_res_list))
    return found_token_index_res_list

def card(id_val, source, context, pdf_html, doc_meta):
    #<div class="card text-white bg-dark mb-3" style="margin:1rem;">
    st.markdown(f"""
    <div class="card" style="margin:1rem;">
        <div class="card-body">
            <h5 class="card-title">{source}</h5>
            <h6 class="card-subtitle mb-2 text-muted">{doc_meta}</h6>
            <p class="card-text">{context}</p>
            <h6 class="card-subtitle mb-2 text-muted">{id_val}</h6>
            {pdf_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def search_query(search_type, post_params, port, api_route, doc_len):
    st.session_state.page = 1
    res = requests.post('http://localhost:{}/{}'.format(port, api_route), json = post_params)
    return res

@st.cache(allow_output_mutation=True)
def load_meta(doc_len):
    port = 5001 + int(doc_len)
    api_route = 'haystack_{}_meta'.format(doc_len)
    res = requests.get('http://localhost:{}/{}'.format(port, api_route))
    return res

@st.cache(allow_output_mutation=True)
def load_wc(wc_text):
    wc_fig = wc.generate_from_text(wc_text)
    return wc_fig

st.set_page_config(layout="wide")

def change_retriever_type():
    if st.session_state['search_type'] == 'by keywords':
        st.session_state['retriever_type'] = 'Exact Match'
    else:
        st.session_state['retriever_type'] = 'Semantic Search (DPR)'

with st.sidebar:
    st.header("Options")
    search_type = st.radio(
        "Search Type:",
        ('by keywords', 'by questions'), key = "search_type", on_change = change_retriever_type)        
    top_k_reader = st.sidebar.slider(
        "Max. number of answers:",
        min_value=1,
        max_value=10,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        key = "top_k_reader"
    )
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever:",
        min_value=1,
        max_value=100,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        key = "top_k_retriever"
    )
    retriever_type = st.radio(
        "Retriever Type:",
        ('Exact Match', 'Semantic Search (DPR)', 'Semantic Search (ER)'),
        key = "retriever_type",
        index = 0,
    )
    doc_len = st.radio(
        "Max. length of documents from retriever:",
        ('100', '200', '300'),
        key = "doc_len",
        index = 1,
    )

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
""", unsafe_allow_html=True)

c01, c02 = st.columns((5, 3))
c01.markdown('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
with c01:
    st.markdown("<div id='linkto_top'></div>", unsafe_allow_html=True)
    st.write("""# RIA Live Demo""")

c11, c12, c13 = st.columns((6, 2, 2))

with c11:
    if st.session_state['search_type'] == 'by questions':
        query = st.text_input('Search ' + st.session_state['search_type'], key = "query", placeholder = "What is the maximum finance charges for credit card?")
    else:
        query = st.text_input('Search ' + st.session_state['search_type'], key = "query")
with c12:
    filter_meta = load_meta(st.session_state['doc_len'])
    filter_central_bank = list(filter_meta.json()['document']['central_bank'].keys())
    filter_keyword = list(filter_meta.json()['document']['keyword'].keys())
    
    filter_central_bank = st.multiselect(
        'Filter Central Banks:',
        filter_central_bank,
        [],
        key = 'filter_central_bank',    
    )
with c13:
    filter_keyword = st.multiselect(
        'Filter Categories:',
        filter_keyword,
        [],
        key = 'filter_keyword',
    )

if query: # or query != '' :
    # with st.spinner(
    #     "🧠 &nbsp;&nbsp; Performing Neural Search on documents... \n "
    # ):
    if search_type == 'by questions':
        post_params = {'query': st.session_state.query, 'top_k_retriever': top_k_retriever, 'top_k_reader': top_k_reader, 
                       'filters': {}, 'retriever_type': retriever_type,
                      }
        if len(st.session_state['filter_central_bank']) > 0:
            post_params['filters']['central_bank'] = st.session_state['filter_central_bank']
        if len(st.session_state['filter_keyword']) > 0:
            post_params['filters']['keyword'] = st.session_state['filter_keyword']
        port = 5001 + int(doc_len)
        api_route = 'haystack_{}_reader_pipe'.format(doc_len)
        res = search_query(search_type, post_params, port, api_route, doc_len)
        st.write("## Results:")
        res_df = pd.DataFrame(res.json()['answers'])
        if len(res_df) > 0:
            res_df['score'] = res_df['score'].astype(float)
            res_df.drop(columns = 'meta', inplace = True)
            content_df = pd.DataFrame(res.json()['documents'])
            content_df.rename(columns = {'id': 'document_id'}, inplace = True)
            res_df = pd.merge(res_df, content_df[['document_id', 'meta', 'content']], on = 'document_id', how = 'left')
            res_df['page'] = res_df.index
            res_df['page'] = res_df['page'] / 10
            res_df['page'] = res_df['page'].astype(int)
            res_df['page'] = res_df['page'] + 1
            res_df['central_bank'] = res_df['meta'].apply(lambda x: x['central_bank'])
            res_df['keyword'] = res_df['meta'].apply(lambda x: x['keyword'])
            res_df['file_name'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-1])
            res_df['full_path'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-4:])
            st.session_state['m ax_page'] = res_df['page'].max()

    elif search_type == 'by keywords':
        post_params = {'query': st.session_state.query, 'top_k_retriever': top_k_retriever, 'top_k_reader': top_k_reader,
                       'filters': {}, 'retriever_type': retriever_type,
                      }
        if len(st.session_state['filter_central_bank']) > 0:
            post_params['filters']['central_bank'] = st.session_state['filter_central_bank']
        if len(st.session_state['filter_keyword']) > 0:
            post_params['filters']['keyword'] = st.session_state['filter_keyword']
        port = 5001 + int(doc_len)
        api_route = 'haystack_{}_retriever_pipe'.format(doc_len)
        res = search_query(search_type, post_params, port, api_route, doc_len)
        res_df = pd.DataFrame(res.json()['documents'])
        if len(res_df) > 0:
            res_df['score'] = res_df['score'].astype(float)
            res_df['page'] = res_df.index
            res_df['page'] = res_df['page'] / 10
            res_df['page'] = res_df['page'].astype(int)
            res_df['page'] = res_df['page'] + 1
            res_df['central_bank'] = res_df['meta'].apply(lambda x: x['central_bank'])
            res_df['keyword'] = res_df['meta'].apply(lambda x: x['keyword'])
            res_df['file_name'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-1])
            res_df['full_path'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-4:])
            st.session_state['max_page'] = res_df['page'].max()

    # Init State Sessioin
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    c21, c22 = st.columns((6, 4))
    
    with c21:
        wc_text = ""
        if len(res_df) > 0:
            filter_res_df = res_df[res_df['page'] == st.session_state['page']]
            for i in range(len(filter_res_df)):
                score = round(filter_res_df['score'].values[i] * 100, 2)
                content = filter_res_df['content'].values[i]
                wc_text = wc_text + " " + content

                if st.session_state['search_type'] == 'by keywords':
                    for each_j in get_found_token(query, content):
                        content = content.replace(each_j, f"<mark>{each_j}</mark>")

                answer = ""
                if 'answer' in list(filter_res_df.columns):
                    answer = filter_res_df['answer'].values[i]
                    content = content.replace(answer,  str(annotation(answer, "ANSWER", "#8ef")))

                doc_meta = "{} | {} | {}".format(filter_res_df['central_bank'].values[i], 
                                                 filter_res_df['keyword'].values[i],
                                                 filter_res_df['file_name'].values[i],
                                                )

                pdf_html = """<a href="http://pc140032646.bot.or.th/pdf/{}/{}/{}/{}" class="card-link">PDF</a> <a href='#linkto_top' class="card-link">Link to top</a> <a href='#linkto_bottom' class="card-link">Link to bottom</a>""".format(filter_res_df['full_path'].values[i][0],filter_res_df['full_path'].values[i][1],filter_res_df['full_path'].values[i][2],filter_res_df['full_path'].values[i][3])

                card('Relevance: {}'.format(score), 
                    answer,
                    '...{}...'.format(content),
                    pdf_html,
                    doc_meta
                )
    with c22:
        if len(wc_text) > 0:
            wc_fig = load_wc(wc_text)
            fig, ax = plt.subplots(figsize = (8, 8))
            ax.imshow(wc_fig, interpolation = 'bilinear')
            plt.axis('off')
            st.pyplot(fig)

    if 'max_page' not in st.session_state:
        st.session_state['max_page'] = 10

    c31, c32 = st.columns((6, 4))
    with c31:
        st.markdown("<div id='linkto_bottom'></div>", unsafe_allow_html=True)
        if int(st.session_state['max_page']) > 1:
            page = st.slider('Page No:', 1, int(st.session_state['max_page']), key = 'page')
            st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)


        


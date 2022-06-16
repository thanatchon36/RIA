ปimport streamlit as st
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

DEFAULT_NUMBER_OF_ANSWERS = 10
DEFAULT_DOCS_FROM_RETRIEVER = 100

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
def search_query(search_type, post_params, api_route, doc_len, distinct_mode):
    st.session_state.page = 1
    port = 5101
    res = requests.post('http://localhost:{}/{}'.format(port, api_route), json = post_params)
    return res

@st.cache(allow_output_mutation=True)
def load_meta():
    port = 6101
    api_route = 'haystack_meta'.format(doc_len)
    res = requests.get('http://localhost:{}/{}'.format(port, api_route))
    return res

@st.cache(allow_output_mutation=True)
def load_wc(wc_text):
    wc_fig = wc.generate_from_text(wc_text)
    return wc_fig

@st.cache(allow_output_mutation=True)
def load_suggested_keywords(post_params):
    port = 5101
    api_route = 'haystack_collocation'
    res = requests.post('http://localhost:{}/{}'.format(port, api_route), json = post_params)
    return res

@st.cache(allow_output_mutation=True)
def load_suggested_questions(post_params):
    port = 5101
    api_route = 'haystack_question_generation'
    res = requests.post('http://localhost:{}/{}'.format(port, api_route), json = post_params)
    return res

st.set_page_config(layout="wide", page_title = 'RIA', page_icon = 'fav.png')

def change_retriever_type():
    if st.session_state['search_type'] == 'by keywords':
        st.session_state['retriever_type'] = 'Exact Match'
    else:
        st.session_state['retriever_type'] = 'Semantic Search (DPR)'

def add_central_bank():
    filter_list = []
    for each in st.session_state['filter_central_bank']:
        temp_list = st.session_state['filter_central_bank_dict'][each]
        filter_list = filter_list + temp_list
    filter_list = list(set(filter_list))
    st.session_state['keyword_list'] = filter_list

    if len(st.session_state['keyword_list']) == 0:
        st.session_state['keyword_list'] = st.session_state['default_keyword_list']

def click_button_sg1():
    st.session_state['query'] = st.session_state['sg_button_1']
def click_button_sg2():
    st.session_state['query'] = st.session_state['sg_button_2']
def click_button_sg3():
    st.session_state['query'] = st.session_state['sg_button_3']
def click_button_sg4():
    st.session_state['query'] = st.session_state['sg_button_4']
def click_button_sg5():
    st.session_state['query'] = st.session_state['sg_button_5']
def click_button_sg6():
    st.session_state['query'] = st.session_state['sg_button_6']
def click_button_sg7():
    st.session_state['query'] = st.session_state['sg_button_7']
def click_button_sg8():
    st.session_state['query'] = st.session_state['sg_button_8']

with st.sidebar:
    st.header("Options")
    search_type = st.radio(
        "Search Type:",
        ('by keywords', 'by questions'), key = "search_type", on_change = change_retriever_type)        
    top_k_reader = st.sidebar.slider(
        "Max. number of answers:",
        min_value=1,
        max_value=20,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        key = "top_k_reader"
    )
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever:",
        min_value=1,
        max_value=200,
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
    distinct_mode = st.radio(
        "Distinct Mode:",
        ('No Distinct', 'Distinct Document', 'Distinct Central Bank'),
        key = "distinct_mode",
        index = 0,
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
    filter_meta = load_meta().json()
    if 'default_central_bank_list' not in st.session_state:
        st.session_state['default_central_bank_list'] = list(filter_meta['central_bank'].keys())
    if 'default_keyword_list' not in st.session_state:
        st.session_state['default_keyword_list'] = list(filter_meta['keyword'].keys())
    if 'central_bank_list' not in st.session_state:
        st.session_state['central_bank_list'] = list(filter_meta['central_bank'].keys())
    if 'keyword_list' not in st.session_state:
        st.session_state['keyword_list'] = list(filter_meta['keyword'].keys())
    if 'filter_central_bank_dict' not in st.session_state:
        st.session_state['filter_central_bank_dict'] = filter_meta['central_bank']
    if 'filter_keyword_dict' not in st.session_state:
        st.session_state['filter_keyword_dict'] = filter_meta['keyword']
    
    filter_central_bank = st.multiselect(
        'Filter Central Banks:',
         st.session_state['central_bank_list'],
        [],
        key = 'filter_central_bank',
        on_change = add_central_bank,
    )
with c13:
    filter_keyword = st.multiselect(
        'Filter Categories:',
        st.session_state['keyword_list'],
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
        api_route = 'haystack_{}_reader_pipe'.format(doc_len)
        res = search_query(search_type, post_params, api_route, doc_len, distinct_mode)
        st.write("## Results:")
        res_df = pd.DataFrame(res.json()['answers'])
        if len(res_df) > 0:
            res_df['score'] = res_df['score'].astype(float)
            res_df.drop(columns = 'meta', inplace = True)
            content_df = pd.DataFrame(res.json()['documents'])
            content_df.rename(columns = {'id': 'document_id'}, inplace = True)
            res_df = pd.merge(res_df, content_df[['document_id', 'meta', 'content']], on = 'document_id', how = 'left')
            res_df['file_name'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-1])
            res_df['central_bank'] = res_df['meta'].apply(lambda x: x['central_bank'])
            res_df['keyword'] = res_df['meta'].apply(lambda x: x['keyword'])
            if distinct_mode == 'Distinct Document':
                res_df = res_df.groupby('file_name').first().reset_index()
            elif distinct_mode == 'Distinct Central Bank':
                res_df = res_df.groupby('central_bank').first().reset_index()
            res_df['page'] = res_df.index
            res_df['page'] = res_df['page'] / 10
            res_df['page'] = res_df['page'].astype(int)
            res_df['page'] = res_df['page'] + 1
            res_df['full_path'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-4:])
            st.session_state['res_df'] = res_df.to_dict()
            st.session_state['max_page'] = res_df['page'].max()

    elif search_type == 'by keywords':
        post_params = {'query': st.session_state.query, 'top_k_retriever': top_k_retriever, 'top_k_reader': top_k_reader,
                       'filters': {}, 'retriever_type': retriever_type,
                      }
        if len(st.session_state['filter_central_bank']) > 0:
            post_params['filters']['central_bank'] = st.session_state['filter_central_bank']
        if len(st.session_state['filter_keyword']) > 0:
            post_params['filters']['keyword'] = st.session_state['filter_keyword']
        api_route = 'haystack_{}_retriever_pipe'.format(doc_len)
        res = search_query(search_type, post_params, api_route, doc_len, distinct_mode)
        st.write("## Results:")
        res_df = pd.DataFrame(res.json()['documents'])
        if len(res_df) > 0:
            res_df['score'] = res_df['score'].astype(float)
            res_df['file_name'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-1])
            res_df['central_bank'] = res_df['meta'].apply(lambda x: x['central_bank'])
            res_df['keyword'] = res_df['meta'].apply(lambda x: x['keyword'])
            if distinct_mode == 'Distinct Document':
                res_df = res_df.groupby('file_name').first().reset_index()
            elif distinct_mode == 'Distinct Central Bank':
                res_df = res_df.groupby('central_bank').first().reset_index()
            res_df['page'] = res_df.index
            res_df['page'] = res_df['page'] / 10
            res_df['page'] = res_df['page'].astype(int)
            res_df['page'] = res_df['page'] + 1
            res_df['full_path'] = res_df['meta'].apply(lambda x: x['file_name'].split('/')[-4:])
            st.session_state['res_df'] = res_df.to_dict()
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
                    for each_j in get_found_token(st.session_state['query'], content):
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

    post_params = {'query': st.session_state.query, 'filters': {}, }
    if len(st.session_state['filter_central_bank']) > 0:
        post_params['filters']['central_bank'] = st.session_state['filter_central_bank']
    if len(st.session_state['filter_keyword']) > 0:
        post_params['filters']['keyword'] = st.session_state['filter_keyword']
    if st.session_state['search_type'] == 'by keywords':
        st.write("#### Suggested Keywords:")
        suggested_keywords = load_suggested_keywords(post_params).json()
    else:
        st.write("#### Suggested Questions:")
        suggested_keywords = load_suggested_questions(post_params).json()
    suggested_list = suggested_keywords['suggested_keywords'][:12]

    c31, c32, cx1 = st.columns((4, 4, 4))
    with c31:
        if len(suggested_list) >= 1:
            sg_button_1 = st.button(suggested_list[0], on_click = click_button_sg1)
            st.session_state['sg_button_1'] = suggested_list[0]
    with c32:
        if len(suggested_list) >= 2:
            sg_button_2 = st.button(suggested_list[1], on_click = click_button_sg2)
            st.session_state['sg_button_2'] = suggested_list[1]

    c33, c34, cx2 = st.columns((4, 4, 4))
    with c33:
        if len(suggested_list) >= 3:
            sg_button_3 = st.button(suggested_list[2], on_click = click_button_sg3)
            st.session_state['sg_button_3'] = suggested_list[2]
    with c34:
        if len(suggested_list) >= 4:
            sg_button_4 = st.button(suggested_list[3], on_click = click_button_sg4)
            st.session_state['sg_button_4'] = suggested_list[3]

    c35, c36, cx3 = st.columns((4, 4, 4))
    with c35:
        if len(suggested_list) >= 5:
            sg_button_5 = st.button(suggested_list[4], on_click = click_button_sg5)
            st.session_state['sg_button_5'] = suggested_list[4]
    with c36:
        if len(suggested_list) >= 6:
            sg_button_6 = st.button(suggested_list[5], on_click = click_button_sg6)
            st.session_state['sg_button_6'] = suggested_list[5]

    c37, c38, cx4 = st.columns((4, 4, 4))
    with c37:
        if len(suggested_list) >= 7:
            sg_button_7 = st.button(suggested_list[6], on_click = click_button_sg7)
            st.session_state['sg_button_7'] = suggested_list[6]
    with c38:
        if len(suggested_list) >= 8:
            sg_button_8 = st.button(suggested_list[7], on_click = click_button_sg8)
            st.session_state['sg_button_8'] = suggested_list[7]

    if 'max_page' not in st.session_state:
        st.session_state['max_page'] = 10
    c41, c42 = st.columns((6, 4))
    with c41:
        st.markdown("<div id='linkto_bottom'></div>", unsafe_allow_html=True)
        if int(st.session_state['max_page']) > 1:
            page = st.slider('Page No:', 1, int(st.session_state['max_page']), key = 'page')
            st.markdown("<a href='#linkto_top'>Link to top</a>", unsafe_allow_html=True)


        


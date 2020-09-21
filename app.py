import logging
from argparse import Namespace
from itertools import permutations

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import yaml

import kss
import torch
import numpy as np

from dash.dependencies import Input, Output, State

from src.common.score.bert_scorer import BertScorer
from src.multi_summary.multisum import MDS
from src.multi_summary.translator import build_predictor as multi_predictor
from src.match_summary.match_selector import build_selector as match_selector
from src.match_summary.matchsum import MatchSum
from src.search.search_handler import Searcher
from src.single_summary.singlesum import Summarization
from src.single_summary.translator import build_predictor as single_predictor

logging.basicConfig(level=getattr(logging, 'INFO'))
logger = logging.getLogger(__name__)

with open('config.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.BaseLoader)

hosts = cfg['elasticsearch']['host']
news_index = cfg['elasticsearch']['news_index']
searcher = Searcher(hosts=hosts, news_index=news_index)

singlesum_model = cfg['singlesum']['model']
singlesum_device = cfg['singlesum']['device']
logger.info("singlesum_model: %s, device: %s", singlesum_model, singlesum_device)
single_summarizer = Summarization.load_from_checkpoint(singlesum_model, map_location=singlesum_device)

single_model = single_summarizer.model.to(singlesum_device)
single_model.eval()

tokenizer = single_summarizer.tokenizer
symbols = {
    "BOS": tokenizer.vocab["<!#s>"],
    "EOS": tokenizer.vocab["<!#/s>"],
    "PAD": tokenizer.vocab["<!#pad>"],
    "PERIOD": tokenizer.vocab["."],
}

multisum_model = cfg['multisum']['model']
multisum_device = cfg['multisum']['device']
logger.info("mulitsum_model: %s, device: %s", multisum_model, multisum_device)
multi_summarizer = MDS.load_from_checkpoint(multisum_model, map_location=multisum_device)

multi_model = multi_summarizer.model.to(multisum_device)
multi_model.eval()

matchsum_model = cfg['matchsum']['model']
matchsum_device = cfg['matchsum']['device']
match_summarizer = MatchSum.load_from_checkpoint(matchsum_model, map_location=matchsum_device)

match_model = match_summarizer.model.to(matchsum_device)
match_model.eval()

single_param = {'alpha': float(cfg['singlesum']['alpha']),
                'block_trigram': cfg['singlesum']['block_trigram'],
                'beam_size': int(cfg['singlesum']['beam_size']),
                'min_length': int(cfg['singlesum']['min_length']),
                'max_length': int(cfg['singlesum']['max_length']),
                }
multi_param = {'alpha': float(cfg['multisum']['alpha']),
               'block_trigram': cfg['multisum']['block_trigram'],
               'beam_size': int(cfg['multisum']['beam_size']),
               'min_length': int(cfg['multisum']['min_length']),
               'max_length': int(cfg['multisum']['max_length']),
               }
match_param = {'min_length': int(cfg['multisum']['min_length']),
               'max_length': int(cfg['multisum']['max_length']),
               }

single_translator = single_predictor(Namespace(**single_param), tokenizer, symbols, single_model)
multi_translator = multi_predictor(Namespace(**multi_param), tokenizer, symbols, multi_model)
match_selector = match_selector(Namespace(**match_param), tokenizer, symbols, match_model)

bert_scorer = BertScorer()


def convert_to_features(docs, device='cuda'):
    features = []
    for doc in docs:
        doc_repr = tokenizer.encode_plus(doc, max_length=512, pad_to_max_length=True, return_tensors="pt")
        doc_repr['input_ids'] = doc_repr['input_ids'].to(device)
        doc_repr['attention_mask'] = doc_repr['attention_mask'].to(device)
        doc_repr['sentence'] = doc
        features.append(doc_repr)
    return features


def singlesum(features, min_len=None, max_len=None):
    for translate in single_translator.translate(features, min_len, max_len):
        pred = tokenizer.decode(translate)

    return pred


def multisum(features: list, min_len=None, max_len=None):
    for translate in multi_translator.translate(features, min_len, max_len):
        pred = tokenizer.decode(translate)

    return pred


def matchsum(features, single_sents, min_len=None, max_len=None, device=matchsum_device):
    permutated_sents = list(permutations(single_sents, 2))
    if permutated_sents:
        first, second = zip(*permutated_sents)
        bert_scores = bert_scorer.score(first, second)

        sorted_sents = [x + " " + y for x, y, _ in sorted(zip(first, second, bert_scores),
                                                          key=lambda pair: pair[2], reverse=True)]
        initial_cands = single_sents + sorted_sents[:10]  # max_value
    else:
        initial_cands = single_sents

    cands = [cand for cand in initial_cands if min_len < len(cand) < max_len]
    logger.info(len(cands))
    logger.info(cands)

    srcs_inpu_ids, srcs_attention_mask = [], []
    for feature in features:
        srcs_inpu_ids.append(feature["input_ids"].squeeze())
        srcs_attention_mask.append(feature["attention_mask"].squeeze())

    cands_sents, cands_input_ids, cands_attention_mask = [], [], []
    for cand in cands:
        encoded_summary = tokenizer.encode_plus(cand, max_length=max_len, pad_to_max_length=True, return_tensors="pt",
                                                device=device)
        cands_sents.append(cand)
        cands_input_ids.append(encoded_summary["input_ids"].squeeze())
        cands_attention_mask.append(encoded_summary["attention_mask"].squeeze())

    srcs = {"input_ids": torch.stack(srcs_inpu_ids),
            "attention_mask": torch.stack(srcs_attention_mask)}
    cands = {"input_ids": torch.stack(cands_input_ids),
             "attention_mask": torch.stack(cands_attention_mask),
             "sents": cands_sents}

    batch = {'srcs': srcs, 'cands': cands}
    return match_selector.select_batches(batch)


search = html.Div(
    [
        dcc.Input(id="input-id", type="text", placeholder="", debounce=True),
        html.Button('Search', id='search-button', n_clicks=0),
        dcc.Dropdown(
            id='category-dropdown',
            options=[
                {'label': 'International Baseball', 'value': 'IBASEBALL'},
                {'label': 'International Soccer', 'value': 'ISOCCER'},
                {'label': 'Baseball', 'value': 'BASEBALL'},
                {'label': 'Entertainment', 'value': 'ENTERTAINMENT'},
                {'label': 'Soccer', 'value': 'SOCCER'},
                {'label': 'Volleyball', 'value': 'VOLLEYBALL'},
                {'label': 'Basketball', 'value': 'BASKETBALL'},
                {'label': 'Golf', 'value': 'GOLF'},
                {'label': 'All', 'value': 'ALL'}
            ],
            value='ALL',
            style={'width': '50%'},
        ),
        html.Div(id='display-value')
    ],
)

slider = html.Div(
    [
        dbc.Label("요약문 길이 (# Tokens)"),
        dcc.RangeSlider(
            id="summary-length",
            min=10,
            max=100,
            step=5,
            value=[30, 80],
            marks={i: str(i) for i in range(10, 101, 5)},
        ),
        dbc.Button('요약', id='summary-button', n_clicks=0)
    ]
)

table_header_style = {
    "backgroundColor": "rgb(2,21,70)",
    "color": "white",
    "textAlign": "center",
}
search_group = html.Div(
    [
        dt.DataTable(
            id="data-table",
            columns=[
                {
                    "name": "Title",
                    "id": "column-title",
                    "type": "text",
                    "selectable": True,
                },
                {
                    "name": "Datetime",
                    "id": "column-datetime",
                    "type": "text",
                    "selectable": False,
                },
                {
                    "name": "Content",
                    "id": "column-content",
                    "type": "text",
                    "selectable": False,
                }
            ],
            editable=False,
            css=[{'selector': 'table', 'rule': 'table-layout: fixed'}],
            style_header=table_header_style,
            style_cell={'textAlign': 'left'},
            style_cell_conditional=[
                {'if': {'column_id': 'column-title'}, 'width': '20%'},
                {'if': {'column_id': 'column-datetime'}, 'width': '10%'},
                {'if': {'column_id': 'column-content'}, 'width': '60%'},
            ],
            row_selectable='multi',
            style_table={'height': '340px', 'overflowY': 'auto', 'width': '100%', 'minWidth': '100%'},
        ),
        dcc.Textarea(
            id='textarea-search',
            value="",
            style={'width': '100%',
                   'font-size': '15px',
                   'height': 250},
        ),
    ]
)


def layout():
    page = html.Div([
        html.H2("연속 담화 내 질의 응대를 위한 다중 문서 분석 시스템"),
        search,
        search_group,
        html.Hr(),
        slider,
        html.Hr(),
        html.Div(
            [
                dbc.Label("단일문서 요약 결과"),
                dcc.Textarea(
                    id='textarea-summary-single',
                    value="",
                    style={'width': '100%',
                           'font-size': '15px',
                           'height': 100},
                ),
                dbc.Label("다중문서 요약(Transformer) 결과"),
                dcc.Textarea(
                    id='textarea-summary-multi',
                    value="",
                    style={'width': '100%',
                           'font-size': '15px',
                           'height': 100},
                ),
                dbc.Label("다중문서 요약(MathcSum) 결과"),
                dcc.Textarea(
                    id='textarea-summary-match',
                    value="",
                    style={'width': '100%',
                           'font-size': '15px',
                           'height': 100},
                )
            ]
        )
    ])
    return page


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'dbc.themes.BOOTSTRAP']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = layout


@app.callback(
    Output('textarea-search', 'value'),
    [
        Input('data-table', 'selected_rows')
    ],
    [
        State('data-table', 'data'),
    ])
def select_news(selected_rows, data):
    content = ""
    if selected_rows:
        row_idx = selected_rows[-1]
        if row_idx < len(data):
            content = data[row_idx]['column-content']

    return content


@app.callback(
    [
        Output('textarea-summary-single', 'value'),
        Output('textarea-summary-multi', 'value'),
        Output('textarea-summary-match', 'value'),
    ],
    [
        Input('summary-button', 'n_clicks'),
        Input('data-table', 'selected_rows'),
        Input('summary-length', 'value')
    ],
    [
        State('data-table', 'data'),
    ]
)
def summary(sum_btn, selected_rows, length_range, data):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    single_summary, multi_summary, match_summary = "", "", ""
    if 'summary-button' in changed_id:
        if selected_rows:
            docs = [data[i]['column-content'] for i in sorted(selected_rows)]
        else:
            docs = [data[i]['column-content'] for i in range(3)]

        if not docs:
            return single_summary, multi_summary, match_summary

        features = convert_to_features(docs)
        single_summaries, single_sents = [], []
        for feature in features:
            pred = singlesum(feature, length_range[0], length_range[1])
            single_summaries.append(pred)
            single_sents.extend(kss.split_sentences(pred))

        single_summary = '\n'.join(single_summaries)
        multi_summary = multisum(features, length_range[0], length_range[1])
        match_summary = matchsum(features, single_sents, length_range[0], length_range[1])

    return single_summary, multi_summary, match_summary


@app.callback(
    [
        Output('data-table', 'data'),
        Output('display-value', 'children')
    ],
    [
        Input('input-id', 'value'),
        Input('category-dropdown', 'value'),
        Input('search-button', 'n_clicks'),
    ]
)
def search_news(query, category, search_btn):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    data, output = [], ""
    if 'search-button' in changed_id:
        if query is not None:
            logger.info("Query: %s, category: %s", query, category)
            responses, count = searcher.search(query, category, from_date="2020-08-31", count=True)
            data = [{"column-title": res['title'], "column-datetime": res['publish_datetime'],
                     "column-content": res['content'].replace("[SEP]", " ").strip()}
                    for res in responses]
            output = '검색된 뉴스 수: {}'.format(count)
    else:
        if query is not None:
            logger.info("Query: %s, category: %s", query, category)
            responses, count = searcher.search(query, category, from_date="2020-08-31", count=True)
            data = [{"column-title": res['title'], "column-datetime": res['publish_datetime'],
                     "column-content": res['content'].replace("[SEP]", " ").strip()}
                    for res in responses]
            output = '검색된 뉴스 수: {}'.format(count)

    return data, output


if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8900, debug=False)

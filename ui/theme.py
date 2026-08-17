"""Tema e estilo do Streamlit."""

import streamlit as st
from PIL import Image


def configurar_pagina():
    icone = Image.open("assets/fototema.jpg")

    st.set_page_config(
        page_title="QuantGuard AI",
        page_icon=icone,
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: #F7F8FC;
        }

        .block-container {
            max-width: 1250px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        h1, h2, h3 {
            color: #171721;
            font-family: Inter, sans-serif;
            letter-spacing: -0.4px;
        }

        p {
            color: #666674;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 28px;
            border-bottom: 1px solid #E5E5EC;
        }

        .stTabs [data-baseweb="tab"] {
            height: 52px;
            background: transparent;
            border: none;
            color: #747481;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: #6C4CE3 !important;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #ECECF2;
            padding: 20px 22px;
            border-radius: 16px;
            box-shadow: 0 5px 20px rgba(20, 20, 40, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #777786;
        }

        div[data-testid="stMetricValue"] {
            color: #171721;
            font-weight: 700;
        }

        div[data-testid="stPlotlyChart"] {
            background: white;
            border: 1px solid #ECECF2;
            border-radius: 18px;
            padding: 10px;
            box-shadow: 0 5px 20px rgba(20, 20, 40, 0.04);
        }

        .audit-card {
            background: white;
            border: 1px solid #ECECF2;
            border-radius: 16px;
            padding: 20px 22px;
            min-height: 150px;
            box-shadow: 0 5px 20px rgba(20, 20, 40, 0.04);
        }

        .audit-label {
            color: #777786;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1px;
        }

        .audit-score {
            color: #171721;
            font-size: 34px;
            font-weight: 800;
            margin: 6px 0;
        }

        .audit-pass {
            color: #16784A;
            font-weight: 700;
        }

        .audit-review {
            color: #9A6B00;
            font-weight: 700;
        }

        .audit-fail {
            color: #B42318;
            font-weight: 700;
        }

        .small-note {
            color: #858594;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_header():
    st.markdown(
        """
<div style="background: linear-gradient(120deg, #171721 0%, #27233A 100%); padding: 38px 42px; border-radius: 22px; margin-bottom: 28px;">

<div style="color: #A991FF; font-size: 13px; font-weight: 600; letter-spacing: 1.6px; margin-bottom: 10px;">
QUANTITATIVE FINANCE × ARTIFICIAL INTELLIGENCE
</div>

<div style="color: white; font-size: 48px; font-weight: 800; letter-spacing: -1.5px;">
QuantGuard AI
</div>

<div style="color: #CBC8D5; font-size: 17px; margin-top: 7px; max-width: 760px; line-height: 1.55;">
A quantitative validation layer designed to benchmark and audit AI-generated financial analysis.
</div>

</div>
        """,
        unsafe_allow_html=True,
    )
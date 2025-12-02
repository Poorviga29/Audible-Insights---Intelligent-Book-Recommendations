import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Audible Insights ",
    layout="wide",
    menu_items=None
)

# ------------------------------------------------------------
# COLUMN DETECTOR
# ------------------------------------------------------------
def detect_col(df, keywords):
    try:
        norm = {re.sub(r"[\s_]+", "", c.lower()): c for c in df.columns}
    except:
        return None

    for kw in keywords:
        k = re.sub(r"[\s_]+", "", kw.lower())
        for nk, actual in norm.items():
            if k in nk:
                return actual

    for c in df.columns:
        for kw in keywords:
            if kw.lower() in c.lower():
                return c
    return None

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data(path="Audible_Insights_Dataset.csv"):
    return pd.read_csv(path)

try:
    df = load_data()
except:
    st.error("❌ Could not load `Audible_Insights_Dataset.csv`.")
    st.stop()

# ------------------------------------------------------------
# DETECT COLUMNS
# ------------------------------------------------------------
title_col = detect_col(df, ["book name", "title"])
author_col = detect_col(df, ["author"])
genre_col = detect_col(df, ["genre"])
rating_col = detect_col(df, ["rating"])
desc_col = detect_col(df, ["description", "summary"])
price_col = detect_col(df, ["price"])
length_col = detect_col(df, ["length", "pages"])
reviews_col = detect_col(df, ["reviews"])
rank_col = detect_col(df, ["rank"])

if title_col is None or genre_col is None:
    st.error("Dataset must include **Book Title** and **Genre** columns.")
    st.stop()

# Clean data
df[title_col] = df[title_col].astype(str)
df[genre_col] = df[genre_col].astype(str)
if author_col:
    df[author_col] = df[author_col].astype(str)
if desc_col:
    df[desc_col] = df[desc_col].astype(str)

numeric_cols = [rating_col, price_col, length_col, reviews_col, rank_col]
for c in numeric_cols:
    if c:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")

# ------------------------------------------------------------
# SIMILARITY MODEL
# ------------------------------------------------------------
@st.cache_resource
def build_similarity(df_local):
    combined = (
        df_local[title_col].fillna("") + " | " +
        df_local[genre_col].fillna("") + " | " +
        (df_local[author_col].fillna("") if author_col else "") + " | " +
        (df_local[desc_col].fillna("") if desc_col else "")
    )

    vect = TfidfVectorizer(stop_words="english", max_features=8000)
    matrix = vect.fit_transform(combined)
    sim = cosine_similarity(matrix, matrix)
    return sim

sim_matrix = build_similarity(df)

# Columns to display
display_cols = [c for c in [
    title_col, author_col, rating_col, reviews_col,
    price_col, desc_col, length_col, rank_col, genre_col
] if c is not None]

# ------------------------------------------------------------
# NEW VISUALIZATION – PRICE vs RATING SCATTER
# ------------------------------------------------------------
def plot_price_scatter(data, title):
    if price_col is None or rating_col is None:
        return None

    valid = data[[price_col, rating_col]].dropna()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid[price_col], valid[rating_col])
    ax.set_xlabel("Price")
    ax.set_ylabel("Rating")
    ax.set_title(title)

    return fig

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
st.sidebar.title("📚 Menu")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏡 Dashboard",
        "📘 Explore Books",
        "🎯 AI Recommendations"
    ]
)

# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------
if page == "🏡 Dashboard":
    st.title("📚 Audible Insights – Intelligent Book Recommendations ")

    # 🌟 NEW: Short project description
    st.markdown("""
    **Audible Insights** is an AI-powered book analysis and recommendation system  
    designed to help users explore books smartly using genre, author, ratings,  
    and intelligent content-based similarity scoring.
    """)

    st.subheader(" Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📚 Total Books", len(df))
    col2.metric("🎭 Total Genres", df[genre_col].nunique())
    col3.metric("🖊 Authors", df[author_col].nunique() if author_col else "-")
    col4.metric("⭐ Avg Rating", f"{df[rating_col].mean():.2f}" if rating_col else "-")

    st.write("---")

    st.subheader("🏆 Top Rated Books")
    st.dataframe(df.sort_values(rating_col, ascending=False).head(5)[display_cols])

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📈 Price vs Rating Scatter")
        fig = plot_price_scatter(df, "Price vs Rating (All Books)")
        if fig:
            st.pyplot(fig)

    with colB:
        st.subheader("⭐ Rating Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(df[rating_col].dropna(), bins=10)
        st.pyplot(fig2)

# ------------------------------------------------------------
# EXPLORE BOOKS
# ------------------------------------------------------------
elif page == "📘 Explore Books":
    st.title("📘 Explore the Library")

    genre_list = sorted(df[genre_col].unique())
    chosen_genre = st.selectbox("🎭 Select Genre", ["All"] + genre_list)

    filtered = df.copy()
    if chosen_genre != "All":
        filtered = filtered[filtered[genre_col] == chosen_genre]

    if author_col:
        authors = sorted(filtered[author_col].unique())
        chosen_author = st.selectbox("🖊 Select Author", ["All"] + authors)
        if chosen_author != "All":
            filtered = filtered[filtered[author_col] == chosen_author]

    st.write(f"### 📚 Showing {len(filtered)} Books")
    st.dataframe(filtered[display_cols])

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📈 Price vs Rating (Filtered)")
        fig = plot_price_scatter(filtered, "Price vs Rating (Filtered Books)")
        if fig:
            st.pyplot(fig)

    with colB:
        st.subheader("⭐ Rating Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(filtered[rating_col].dropna(), bins=10)
        st.pyplot(fig2)

# ------------------------------------------------------------
# RECOMMENDATION PAGE
# ------------------------------------------------------------
elif page == "🎯 AI Recommendations":
    st.title("🎯 AI Book Recommendations")

    selected_book = st.selectbox("📖 Choose a Book", df[title_col].unique())
    index = df.index[df[title_col] == selected_book][0]

    scores = sorted(list(enumerate(sim_matrix[index])), key=lambda x: x[1], reverse=True)
    recommended = df.iloc[[i for i, _ in scores[1:6]]].copy()
    recommended["similarity"] = [round(s, 3) for _, s in scores[1:6]]

    st.subheader(" Recommended Books")
    st.dataframe(recommended[display_cols] + ["similarity"])

    colA, colB = st.columns(2)

    with colA:
        st.subheader("📈 Price vs Rating (Recommended)")
        fig = plot_price_scatter(recommended, "Price vs Rating – Recommended Books")
        if fig:
            st.pyplot(fig)

    with colB:
        st.subheader("⭐ Rating Spread")
        fig2, ax2 = plt.subplots()
        ax2.hist(recommended[rating_col].dropna(), bins=5)
        st.pyplot(fig2)

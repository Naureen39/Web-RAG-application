import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# --- Setup
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
LLM_model = "meta-llama/llama-4-maverick-17b-128e-instruct"

# --- LLM Function for Web Page Analysis
def generate_answer_from_url(llm_model, page_text, user_query):
    llm = ChatGroq(
        model=llm_model,
        temperature=0.2,
        max_tokens=1024,
        timeout=None,
        max_retries=2,
    )

    prompt = f"""
You are a helpful assistant that answers user questions based on the content of a web page.
Use the information from the page to provide a concise and informative response to the user's question.
Do not include any irrelevant details. If the answer isn't in the content, respond accordingly.

[Page Content]: {page_text}  # Limit to first 4000 chars

[User Question]: {user_query}
Instruct: 
- Answer the question based on the page content. Do not say you don't know or page does not contain information.
- Try to formulate the answer with best of your capabilities. Be concise and informative.
"""

    messages = [
        ("system", "You are a helpful web-reading assistant."),
        ("human", prompt),
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content

# --- LLM Function for Book Data Analysis
def generate_book_analysis(LLM_model, book_data_csv, price_threshold):
    llm = ChatGroq(
        model=LLM_model,
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prompt = f""" \
You are a data analyst reviewing the results of a book scraping project.
The data contains book titles, prices in GBP, and availability status.
The analysis is filtered for books under £{price_threshold}. Below is a sample of the data.

Your task:
- Summarize the overall price distribution.
- Comment on common availability statuses.
- Highlight any interesting patterns or anomalies.
- Keep the analysis concise (1–2 paragraphs max).

Input Data:
{book_data_csv}
"""

    messages = [
        ("system", "You are a concise and intelligent data analyst."),
        ("human", prompt),
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content

# --- Streamlit UI
st.set_page_config(page_title="Web Scraper & RAG Assistant", layout="centered")
st.title("📚 Web Scraper & RAG Assistant")
st.markdown("Scrape books and analyze data using AI. You can also ask questions based on page content!")

# --- Scraper Settings
st.sidebar.subheader("Web Scraping Parameters")
num_pages = st.sidebar.slider("Number of pages to scrape", 1, 5, 1)
max_price = st.sidebar.slider("Max price (£)", 5.0, 60.0, 20.0)

# --- Web Scraping Function
@st.cache_data(show_spinner=True)
def scrape_books(pages, max_price):
    BASE_URL = "http://books.toscrape.com/catalogue/page-{}.html"
    books = []

    for page_num in range(1, pages + 1):
        url = BASE_URL.format(page_num)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        book_list = soup.select("article.product_pod")
        for book in book_list:
            title = book.h3.a['title']
            price_text = book.select_one('.price_color').text.strip()
            availability = book.select_one('.availability').text.strip()
            price = float(re.sub(r'[^\d.]', '', price_text))

            if price <= max_price:
                books.append({
                    "Title": title,
                    "Price (£)": price,
                    "Availability": availability
                })

    return pd.DataFrame(books)

# --- UI for Book Scraping and Analysis
if st.button("Scrape Books"):
    with st.spinner("Scraping..."):
        df = scrape_books(num_pages, max_price)
        st.session_state["book_data"] = df  # Save data in session state

# --- Show Data if Already Scraped
if "book_data" in st.session_state:
    df = st.session_state["book_data"]
    if df.empty:
        st.warning("No books found under the selected price range.")
    else:
        st.success(f"Found {len(df)} books under £{max_price}")
        st.dataframe(df)

        # Price Distribution
        st.subheader("📊 Price Distribution")
        fig, ax = plt.subplots()
        df["Price (£)"].hist(bins=10, color="skyblue", edgecolor="black", ax=ax)
        ax.set_xlabel("Price (£)")
        ax.set_ylabel("Number of Books")
        st.pyplot(fig)

        # CSV Download
        csv = df.to_csv(index=False).encode("utf-8")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("📥 Download CSV", csv, f"books_under_{max_price}_{timestamp}.csv", "text/csv")

        # LLM Summary
        if st.button("🔍 LLM Summary"):
            st.subheader("📑 AI Summary of Findings")
            with st.spinner("Analyzing with LLM..."):
                sample_csv = df.head(10).to_csv(index=False)
                llm_summary = generate_book_analysis(LLM_model, sample_csv, max_price)
            st.write(llm_summary)

# --- Web-based RAG Section
st.sidebar.subheader("Ask Questions on Web Page")
url = st.sidebar.text_input("Enter a URL:")
question = st.sidebar.text_area("Ask your question based on the page content:")

if st.sidebar.button("Answer My Question"):
    if not url or not question:
        st.warning("Please enter both a URL and a question.")
    else:
        with st.spinner("Fetching and analyzing page..."):
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)

                answer = generate_answer_from_url(LLM_model, text, question)
                st.subheader("📘 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Failed to fetch or process the page: {e}")
        st.success("Done!")

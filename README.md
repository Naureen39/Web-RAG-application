# Web Rag Assistant

## Overview
Web Rag Assistant is a Python-based application designed to automate web scraping tasks and provide AI-powered analysis of scraped data. It includes features for extracting book data, analyzing price distributions, and answering user questions based on web page content.

## Features
- Scrape book data from websites with customizable parameters (e.g., number of pages, price range).
- Generate AI-powered summaries of scraped data using LLMs.
- Answer user questions based on the content of web pages.
- Interactive Streamlit-based user interface.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Naureen39/Web-RAG-application.git
   ```
2. Navigate to the project directory:
   ```bash
   cd web_scrapping
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   streamlit run web_rag_assistant.py
   ```
2. Use the sidebar to configure scraping parameters:
   - Select the number of pages to scrape.
   - Set the maximum price for books.
3. Click the "Scrape Books" button to start scraping.
4. View the scraped data, analyze price distributions, and download the results as a CSV file.
5. Use the AI-powered summary feature to analyze the scraped data.
6. Enter a URL and a question in the sidebar to get AI-generated answers based on web page content.

## Environment Variables
Create a `.env` file in the project directory and add the following:
```
GROQ_API_KEY=<your_groq_api_key>
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License
This project is licensed under the MIT License.

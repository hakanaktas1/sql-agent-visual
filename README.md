# SQL Agent with Visualization & Streamlit UI 📊

This project implements an intelligent AI Agent capable of querying a SQL database (`olist.sqlite`) and generating data visualizations (charts/graphs) on demand. It features a professional **Streamlit** web interface for user interaction.

## 🚀 Features

*   **Streamlit Web UI**: A clean, corporate-style chat interface for interacting with the agent.
*   **Text-to-SQL**: Converts natural language questions into valid SQL queries for the Olist E-commerce dataset.
*   **Data Visualization**: Automatically generates bar charts, line graphs, etc., using `matplotlib` when requested.
*   **Intelligent Routing**: Uses **LangGraph** to decide when to query SQL, when to check schemas, and when to plot data.
*   **Model**: Powered by **Google Gemini 2.5 Flash** (via `langchain-google-genai`).

## 🏗️ Architecture

The agent uses a ReAct-style loop to reason about data and visualization:

![Agent Architecture](agent_architecture.png)

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/hakanaktas1/sql-agent-visual.git
    cd sql-agent-visual
    ```

2.  Install dependencies:
    ```bash
    pip install langchain langchain-google-genai langchain-community langchain-experimental matplotlib streamlit
    ```

## 🔑 Configuration

You need a **Google Gemini API Key**.

1.  Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY=AIzaSy...
    ```

## 🏃‍♂️ Usage

### 1. Web Interface (Recommended)
Run the Streamlit app:
```bash
python -m streamlit run app.py
```

### 2. Terminal Mode
Run the backend script directly:
```bash
python sql_agent_visual.py
```

### Example Queries (Turkish Supported)
*   "En çok satılan ürün kategorilerini listele ve bar chart çiz"
*   "Hangi eyaletten (customer_state) en çok sipariş geliyor?"
*   "Show me the top 5 sellers by revenue."

## 📂 Project Structure

*   `app.py`: Streamlit frontend application.
*   `sql_agent_visual.py`: Main agent logic and LangGraph definition.
*   `olist.sqlite`: E-commerce dataset.
*   `agent_architecture.png`: Visual representation of the agent's logic.

## 📝 License

This project is open source.

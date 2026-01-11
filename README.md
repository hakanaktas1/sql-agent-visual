# SQL Agent with Visualization 📊

This project implements an intelligent AI Agent capable of querying a SQL database (`Chinook.db`) and generating data visualizations (charts/graphs) on demand.

It uses **LangChain**, **LangGraph**, and **OpenAI/Google Gemini** models via **OpenRouter** to process natural language queries, execute SQL, and run Python code for plotting.

## 🚀 Features

*   **Text-to-SQL**: Converts natural language questions into valid SQL queries.
*   **Data Visualization**: Automatically generates bar charts, line graphs, etc., using `matplotlib` when requested.
*   **Self-Correction**: Can check table schemas and retry queries if errors occur.
*   **Cost-Effective**: Configured to use efficient models like `Gemini 1.5 Flash` via OpenRouter.

## 🛠️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/hakanaktas1/sql-agent-visual.git
    cd sql-agent-visual
    ```

2.  Install dependencies:
    ```bash
    pip install langchain langchain-openai langchain-community langchain-experimental matplotlib seaborn
    ```

## 🔑 Configuration

You need an **OpenRouter API Key** to run the agent.

1.  Open `sql_agent_visual.py`.
2.  Find the following line and replace it with your key (or set it as an environment variable):
    ```python
    os.environ["OPENROUTER_API_KEY"] = "sk-or-..." 
    ```

## 🏃‍♂️ Usage

Run the agent with the following command:

```bash
python sql_agent_visual.py
```

### Example Queries
*   "Show me the top 5 artists by number of tracks and plot a bar chart."
*   "How many invoices are there per country? Visualize it."
*   "List the tracks in the 'Rock' genre."

## 📂 Project Structure

*   `sql_agent_visual.py`: The main script containing the Agent logic and Graph definition.
*   `Chinook.db`: Sample SQLite database (Music Store data).

## 📝 License

This project is open source.

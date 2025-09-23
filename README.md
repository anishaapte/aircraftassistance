# aircraftassistance
RAG based aircraft assistance

# Pre-Requisites
- Create .env file with the following
<pre>
DATABASE_URL=your-db-url
API_BASE=https://your-api-base-url
API_KEY=your-secret-api-key
MODEL_NAME=your-model-name
OPENAI_API_KEY=your-api-key

</pre>
- This rag application uses pgvector database. Use your database connection string as the database url. 

# Run Jupyter Lab notebook
- Install uv with the official guide https://docs.astral.sh/uv/getting-started/installation/
- Clone the github repository
- project already has pyproject.toml file
<pre>
  uv sync
</pre>
- Launch Jupyter notebook with uv
<pre>
  uv run --with jupyter jupyter lab
</pre>
- Open the rag_pipeline.ipynb file
- Enter your db_uri and open_api_key

# Run Chatbot assistant
- python ragapp.py

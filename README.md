# aircraftassistance
RAG based aircraft assistance

# to run Juoyter Lab notebook
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

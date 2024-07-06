#Need to add the following line to the Dockerfile
FROM ollama/ollama

RUN ["ollama","pull", "nomic-embed-text"]

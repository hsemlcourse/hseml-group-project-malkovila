FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN python -m nltk.downloader -d /usr/local/share/nltk_data \
        punkt averaged_perceptron_tagger stopwords vader_lexicon

ENV NLTK_DATA=/usr/local/share/nltk_data

COPY . .

CMD ["bash"]

import types

import app


class DummyIndex:
    def __init__(self, scores, indices):
        self._scores = scores
        self._indices = indices

    def search(self, emb, k):
        # ignore emb and k, return preconfigured arrays
        import numpy as np

        return (np.array([self._scores], dtype="float32"),
                np.array([self._indices], dtype="int64"))


class DummyMongo:
    def __init__(self, docs_by_idx):
        self._docs_by_idx = docs_by_idx

    def find(self, query):
        # query is like {"faiss_idx": {"$in": ids}}
        ids = query["faiss_idx"]["$in"]
        for i in ids:
            if i in self._docs_by_idx:
                yield self._docs_by_idx[i]


class DummyEmbedder:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        # Return a 1xD vector; contents don't matter because DummyIndex ignores it
        return np.zeros((1, 4), dtype="float32")


def test_retrieve_returns_ranked_results(monkeypatch):
    # Prepare dummy index and mongo
    app.index = DummyIndex(scores=[0.9, 0.5], indices=[0, 1])
    app.mongo = DummyMongo(
        {
            0: {"faiss_idx": 0, "text": "First document"},
            1: {"faiss_idx": 1, "text": "Second document"},
        }
    )
    app.embedder = DummyEmbedder()

    results = app.retrieve("test query", k=2)
    assert len(results) == 2
    assert results[0]["text"] == "First document"
    assert results[0]["score"] >= results[1]["score"]


def test_rag_idk_when_no_context(monkeypatch):
    # Force retrieve to return empty list
    monkeypatch.setattr(app, "retrieve", lambda q, k=3: [])

    out = app.rag("anything")
    assert out == app.IDK_MESSAGE


def test_rag_calls_ollama_when_confident(monkeypatch):
    # Force retrieve to return one good context with high score
    monkeypatch.setattr(
        app,
        "retrieve",
        lambda q, k=3: [{"text": "Some context", "score": app.STRICT_REFUSAL_THRESHOLD + 0.1}],
    )

    # Dummy ollama.chat that returns a predictable answer
    class DummyResponse:
        message = {"content": "Hello from test model"}

    def dummy_chat(model, messages):
        return {"message": {"content": "Hello from test model"}}

    monkeypatch.setattr(app, "ollama", types.SimpleNamespace(chat=dummy_chat))

    out = app.rag("question")
    assert "Hello from test model" in out

import pytest

from rag_qa.embeddings import EmbeddingError, GeminiEmbedder


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_embedder_batches_and_task_types(monkeypatch):
    # create fake response objects
    emb = lambda vals: Obj(values=vals)
    resp = Obj(embeddings=[emb([0.1, 0.2]), emb([0.3, 0.4])])
    resp_q = Obj(embeddings=[emb([0.5, 0.6])])
    client = FakeClient([resp, resp_q])

    e = GeminiEmbedder(api_key="x", model="m", dimensions=2, batch_size=2)
    e._client = client
    out = e.embed_documents(["a", "b"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    q = e.embed_query("q")
    assert q == [0.5, 0.6]
    assert client.models.calls[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert client.models.calls[1]["config"].task_type == "RETRIEVAL_QUERY"


def test_embedder_dimension_mismatch():
    emb = lambda vals: Obj(values=vals)
    resp = Obj(embeddings=[emb([0.1, 0.2, 0.3])])
    client = FakeClient([resp])
    e = GeminiEmbedder(api_key="x", model="m", dimensions=2)
    e._client = client
    with pytest.raises(EmbeddingError, match="dimension mismatch"):
        e.embed_documents(["a"])


def test_embedder_retry_and_redaction(monkeypatch):
    calls = {"n": 0}

    class FlakyModels:
        def embed_content(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("429 rate limit for key SECRET123")
            return Obj(embeddings=[Obj(values=[0.1, 0.2])])

    class C:
        models = FlakyModels()

    e = GeminiEmbedder(api_key="SECRET123", model="m", dimensions=2, max_retries=3)
    e._client = C()
    monkeypatch.setattr("time.sleep", lambda *_: None)
    out = e.embed_documents(["a"])
    assert out == [[0.1, 0.2]]
    assert calls["n"] == 2

    # non-retryable error should not leak key
    class BadModels:
        def embed_content(self, **kwargs):
            raise RuntimeError("bad request with SECRET123")

    class C2:
        models = BadModels()

    e2 = GeminiEmbedder(api_key="SECRET123", model="m", dimensions=2, max_retries=1)
    e2._client = C2()
    with pytest.raises(EmbeddingError) as exc:
        e2.embed_documents(["a"])
    assert "SECRET123" not in str(exc.value)

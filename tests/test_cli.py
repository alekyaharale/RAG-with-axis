import types

from rag_qa.cli import _extract_final_text


class E:
    def __init__(self, text=None):
        if text is None:
            self.content = None
        else:
            self.content = types.SimpleNamespace(parts=[types.SimpleNamespace(text=text)])


def test_extract_final_text():
    events = [E("first"), E(None), E("final answer")]
    assert _extract_final_text(events) == "final answer"
    assert _extract_final_text([]) == ""

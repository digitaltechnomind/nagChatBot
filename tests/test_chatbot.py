import pytest
from app.rag_chatbot import RAGChatbot
from tests.test_data import TEST_QUESTIONS

@pytest.fixture(scope="module")
def chatbot():
    return RAGChatbot()

def test_full_conversation(chatbot):
    responses = []

    for q in TEST_QUESTIONS:
        response, _ = chatbot.ask(q)
        responses.append(response)

    assert len(responses) == 10


def test_out_of_context(chatbot):
    response, _ = chatbot.ask("What is the capital of France?")
    assert "not available" in response.lower()


def test_response_not_empty(chatbot):
    response, _ = chatbot.ask("Explain embeddings")
    assert len(response.strip()) > 10
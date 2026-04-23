from app.rag_chatbot import RAGChatbot

def test_retrieval_returns_docs():
    bot = RAGChatbot()

    _, docs = bot.ask("Explain embeddings")

    assert len(docs) > 0
    assert docs[0].page_content is not None


def test_retrieval_relevance():
    bot = RAGChatbot()

    _, docs = bot.ask("vector embeddings")

    combined = " ".join([d.page_content.lower() for d in docs])

    assert "embedding" in combined
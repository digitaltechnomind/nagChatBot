from app.rag_chatbot import RAGChatbot

def test_sliding_window_memory():
    bot = RAGChatbot()

    for i in range(6):
        bot.ask(f"Question {i}")

    assert len(bot.chat_history) == 4


def test_memory_context():
    bot = RAGChatbot()

    bot.ask("What is AI?")
    bot.ask("Explain ML")
    response, _ = bot.ask("What did I ask first?")

    assert "AI" in response or "ai" in response.lower()
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

class RAGChatbot:
    def __init__(self, index_path="faiss_pdfindex"):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.db = FAISS.load_local(
            index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.chat_history = []

    def _format_history(self):
        formatted = ""
        for q, a in self.chat_history[-4:]:
            formatted += f"User: {q}\nAssistant: {a}\n"
        return formatted.strip()

    def _get_chain(self):
        template = """
        Answer using context and chat history.

        If not found:
        "answer is not available in the context"

        Chat History:
        {chat_history}

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        model = Ollama(model="llama3")
        return prompt | model

    def ask(self, question):
        docs = self.db.similarity_search(question)
        context = "\n\n".join([d.page_content for d in docs])

        chain = self._get_chain()

        response = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": self._format_history()
        })

        self.chat_history.append((question, response))
        self.chat_history = self.chat_history[-4:]

        return response, docs
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Load OpenAI embedding and FAISS vectorstore
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = FAISS.load_local(
    "ticket_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Define the function (same as in your code)
def find_similar_tickets(query: str, llm: ChatOpenAI, top_k: int = 3):
    print("[SimilarityChecker] Finding similar tickets...")
    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    similar_results = []

    for doc in retrieved_docs:
        content = doc.page_content
        prompt = f"""
        You're a helpful support assistant. Given a new ticket description:

        NEW TICKET:
        {query}

        PAST TICKET:
        {content}

        Does the past ticket describe a similar issue? Reply only "Yes" or "No".
        """
        result = llm.invoke([HumanMessage(content=prompt)])
        response = result.content.strip().lower()
        print(f"\nLLM Response: {response}")  # Optional for debugging
        if "yes" in response:
            similar_results.append(content)

    return similar_results

# Initialize the LLM
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 🔍 Try a test ticket
query = "Would you be able to help correct the whatever the issue is with the posting of PI batch 7268 please?"
similar = find_similar_tickets(query, llm, top_k=3)

# 🖨️ Print the similar tickets
print("\n=== SIMILAR TICKETS FOUND ===")
for i, ticket in enumerate(similar, 1):
    print(f"\n--- Ticket #{i} ---\n{ticket}")

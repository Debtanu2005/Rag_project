from langchain.agents import create_agent
from pydantic import BaseModel, Field

class RAGAnswer(BaseModel):
    answer: str = Field(description="Answer based only on the retrieved context")

answer_agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[],
    response_format=RAGAnswer,
    system_prompt="""You are a RAG question-answering assistant.
Answer the user's query using only the provided context.
Do not use external knowledge.
If the context does not contain enough information, say:
'I don't have enough information in the retrieved documents.'
Keep the answer clear and concise."""
)

def answer_query(query: str, documents) -> str:
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    result = answer_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""
    Context:
    {context}

    Question:
    {query}

    Answer using only the context.
    """
                }
            ]
        })

    return result["structured_response"].answer
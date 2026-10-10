

from pydantic import BaseModel, Field
from langchain.agents import create_agent

class QueryGrade(BaseModel):
    relevant: str = Field(description="Return only Yes if the query is a meaningful question related to the application's domain, otherwise No")

agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[],
    response_format=QueryGrade
)

def grade_query(query: str) -> str:
    result = agent.invoke({
        "messages": [
            {
                "role": "system",
                "content": "Classify the user query. Return Yes only for a meaningful question related to the application's domain. Return No for greetings like hi, hello, thanks, or unrelated messages. Output only Yes or No."
            },
            {"role": "user", "content": query}
        ]
    })
    return result["structured_response"].relevant
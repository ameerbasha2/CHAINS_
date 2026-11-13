from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch,RunnableLambda
llm=ChatOllama(model="qwen2.5:1.5b")
class Feedback(BaseModel):
    sentiment:Literal["Positive","Negative"]=Field(description="give the feed back of the review")
parser1=StrOutputParser()
parser2=PydanticOutputParser(pydantic_object=Feedback)
prompt1=PromptTemplate(
    template="classify the sentiment for the respective feed back of {feedback}.\n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions":parser2.get_format_instructions()}
)
classifer_chain=prompt1|llm|parser2
prompt2=PromptTemplate(
    template="Write an appropriate response to this positive feedback {feedback}",
    input_variables=["feedback"]
)
prompt3=PromptTemplate(
    template="Write an appropriate response to this negative feedbck{feedback}",
    input_variables=["feedback"]
)
branch_chain=RunnableBranch(
    (lambda x:x.sentiment=="Positive",prompt2|llm|parser1),
    (lambda x:x.sentiment=="Negative",prompt3|llm|parser1),
    RunnableLambda(lambda x:"could not find the sentiment")
)
chain=classifer_chain|branch_chain
result=chain.invoke({"feedback":"this is a worst mobile that i had sceen in my life"})
print(result)

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
llm=ChatOllama(model="qwen2.5:1.5b")
template1=PromptTemplate(
    template="henerate the 4 interesting facts about the  {topic}",
    input_variables=["topic"]
)
parser=StrOutputParser()
chain=template1|llm|parser
prompt=chain.invoke({"topic":"CRICKET"})
print(prompt)
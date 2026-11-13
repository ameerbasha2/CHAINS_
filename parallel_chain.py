
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


llm = ChatOllama(model="qwen2.5:1.5b")
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question-answer pairs from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""Merge the provided notes and quiz into a single document.Notes:{notes}Quiz:{quiz}""",
    input_variables=["notes", "quiz"]
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt1 | llm | parser,
    "quiz": prompt2 | llm | parser
})
merge_chain = prompt3 | llm | parser
chain = parallel_chain | merge_chain
text = """
Support vector machines (SVMs) are a set of supervised learning
methods used for classification, regression and outlier detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where the number of dimensions is greater
than the number of samples.

Uses a subset of training points in the decision function
(called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the
decision function.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples,
avoiding over-fitting is important.

SVMs do not directly provide probability estimates.

The support vector machines in scikit-learn support both dense
and sparse sample vectors as input.
"""
result = chain.invoke({"text": text})
print(result)
chain.get_graph().print_ascii()

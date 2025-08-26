from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_retrieval_chain(vs, llm, template: str):
    """Creates a retrieval chain."""
    prompt = PromptTemplate.from_template(template)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={'k': 5})
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain
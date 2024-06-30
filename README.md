# Notebook for RAG Over LangChain Documentation

This notebook demonstrates the usage of LangChain for creating a retrieval-based question-answering system. It includes the setup of necessary dependencies, environment variables, and the LangChain components for document retrieval and response generation.

## Setup

### Install Dependencies

To install the required dependencies, run the following commands:

```bash
pip install -U langchain langsmith langchainhub langchain_benchmarks
pip install chromadb openai huggingface pandas langchain_experimental sentence_transformers pyarrow anthropic tiktoken
```

### Set Environment Variables

Update these environment variables with your own API keys:

```python
import os

os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"

os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Silence warnings from HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Generate Unique Run ID

Generate a unique run ID for this experiment:

```python
import uuid

run_uid = uuid.uuid4().hex[:6]
```

## Clone Public Dataset

Clone the LangChain public dataset:

```python
from langchain_benchmarks import clone_public_dataset, registry
from langsmith import traceable

# Filter the registry to include only RetrievalTasks
registry = registry.filter(Type="RetrievalTask")

# Get the LangChain Docs Q&A dataset
langchain_docs = registry["LangChain Docs Q&A"]

# Clone the dataset
clone_public_dataset(langchain_docs.dataset_id, dataset_name=langchain_docs.name)
```

## Retrieve Documents

Retrieve documents from the LangChain dataset:

```python
docs = list(langchain_docs.get_docs())
print(repr(docs[0])[:100] + "...")
```

## Create Embeddings and Vector Store

Create embeddings and a vector store using HuggingFace models:

```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

vectorstore = Chroma(
    collection_name="lcbm-b-huggingface-gte-base",
    embedding_function=embeddings,
    persist_directory="./chromadb",
)

vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
```

## Set Up the Response Chain

Set up the response chain using LangChain components:

```python
from operator import itemgetter
from typing import Sequence
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable.passthrough import RunnableAssign

@traceable
def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = (
            f"<document index='{i}'>\n"
            f"<source>{doc.metadata.get('source')}</source>\n"
            f"<doc_content>{doc.page_content}</doc_content>\n"
            "</document>"
        )
        formatted_docs.append(doc_string)
    formatted_str = "\n".join(formatted_docs)
    return f"<documents>\n{formatted_str}\n</documents>"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant answering questions about LangChain."
            "\n{context}\n"
            "Respond solely based on the document content.",
        ),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

response_generator = (prompt | llm | StrOutputParser()).with_config(
    run_name="GenerateResponse",
)

chain = (
    RunnableAssign(
        {
            "context": (itemgetter("question") | retriever | format_docs).with_config(
                run_name="FormatDocs"
            )
        }
    )
    | response_generator
)
```

## Query the Model

Invoke the chain with a sample question:

```python
response = chain.invoke({"question": "Tell me how a chain works in LangChain. In 3 sentences."})
print(response)
```

## Notes

- Ensure that you have set your API keys correctly.
- The `traceable` decorator from `langsmith` is used to monitor the performance of the functions.
- The `warn_deprecated` warnings can be ignored as they are due to deprecations in the LangChain library.

from io import BytesIO
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from litellm import completion
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": None,
            "question": "",
            "doc_text": "",
            "retrieved_chunks": [],
            "error": None,
        },
    )


def extract_text_from_upload(file: UploadFile, raw_bytes: bytes) -> str:
    filename = (file.filename or "").lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages).strip()

    try:
        return raw_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="ignore").strip()


def build_retriever_from_text(doc_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(doc_text)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever, chunks


@app.post("/", response_class=HTMLResponse)
async def ask_document(
    request: Request,
    question: str = Form(...),
    document: UploadFile = File(...),
):
    try:
        raw_bytes = await document.read()
        doc_text = extract_text_from_upload(document, raw_bytes)

        if not doc_text:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "answer": None,
                    "question": question,
                    "doc_text": "",
                    "retrieved_chunks": [],
                    "error": "Could not extract any text from the uploaded document.",
                },
            )

        retriever, _ = build_retriever_from_text(doc_text)
        docs = retriever.invoke(question)
        retrieved_chunks = [doc.page_content for doc in docs]

        context = "\n\n---\n\n".join(retrieved_chunks)

        response = completion(
            model=os.getenv("LITELLM_MODEL", "openai/gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Answer only from the retrieved document context. "
                        "If the answer is not in the context, say you cannot find it in the document."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Retrieved document context:
{context}

Question:
{question}

Answer based only on the retrieved document context above.
""",
                },
            ],
        )

        answer = response.choices[0].message.content

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": answer,
                "question": question,
                "doc_text": doc_text[:1000],
                "retrieved_chunks": retrieved_chunks,
                "error": None,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "answer": None,
                "question": question,
                "doc_text": "",
                "retrieved_chunks": [],
                "error": f"Error: {str(e)}",
            },
        )
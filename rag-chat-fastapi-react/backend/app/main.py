from fastapi import FastAPI, File as FastAPIFile, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import shutil
from pathlib import Path
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel
import asyncio
import os
from transformers import AutoTokenizer, AutoModel
from .rag.rag import get_chunks, document_map_embedding, query_compute_embeddings, select_top_k_chunks, retreive_chunks_content, generate_llm_response
from .db.database import File as DBFile, get_db, DATABASE_URL
import openai
import json


# Directory to store uploaded files
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Define FastAPI app
app = FastAPI()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Allow CORS for React frontend on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (uploaded files)
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# Name of the model
model_name = "BAAI/bge-small-en-v1.5"

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Model initialization
model = AutoModel.from_pretrained(model_name)


@app.post("/send_message_and_upload/")
async def send_message_and_upload(
    message: str = Form(...),
    file: Optional[UploadFile] = FastAPIFile(None),
    db: Session = Depends(get_db),
):
    doc_chunks = {}
    file_info = None
    if file:
        
        safe_filename = Path(file.filename).name
        file_location = UPLOAD_DIR / safe_filename

        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_extension = safe_filename.split('.')[-1]
        file_summary = ''
        upload_date = datetime.fromtimestamp(file_location.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        # Query using the aliased DBFile model
        existing_file = db.query(DBFile).filter(DBFile.filename == safe_filename).first()

        if not existing_file:
            db_file = DBFile(
                filename=safe_filename,
                extension=file_extension,
                file_location=str(file_location),
                upload_date=upload_date,
                summary=file_summary,
                status='Not started',
                chunks=json.dumps([]),
                take_into_account='Disable'
            )
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
            current_record = db_file
        else:
            
            current_record = existing_file

        file_info = {
            "filename": current_record.filename,
            "extension": current_record.extension,
            "upload_date": current_record.upload_date,
            "file_location": current_record.file_location,
            "file_url": f"http://localhost:8000/static/{current_record.filename}",
            "summary": current_record.summary,
            "status": current_record.status,
            "chunks": json.loads(current_record.chunks) if current_record.chunks else [],
            "take_into_account": current_record.take_into_account,
        }

    # Query only files where take_into_account is set to 'Enable'
    files = db.query(DBFile).filter(DBFile.take_into_account == "Enable").all()
    relevant_chunks = None
    if files:
        for f_item in files:
            if f_item.chunks:
                chunks_list = json.loads(f_item.chunks)
                doc_chunks[f_item.filename] = {chunk["chunk_id"]: {"text": chunk["chunk_text"]} for chunk in chunks_list}

        if doc_chunks:
            query_embeddings = query_compute_embeddings(message, tokenizer, model)

            all_chunks_embeddings = {}
            for filename, chunks in doc_chunks.items():
                all_chunks_embeddings[filename] = document_map_embedding(chunks, tokenizer, model)

            top_k_result = select_top_k_chunks(query_embeddings, all_chunks_embeddings, top_k=3)
            relevant_chunks = retreive_chunks_content(top_k_result, doc_chunks)

    bot_response = generate_llm_response(message, relevant_chunks)

    return JSONResponse(content={
        "message": bot_response,
        "file": file_info
    })


class FileLocation(BaseModel):
    filename: str
    file_location: str


async def process_the_file(
        file_path: str,
        filename: str,
        db: Session):

    doc_chunks = get_chunks(file_path, tokenizer)

    if doc_chunks:
        chunks_list = [{"chunk_id": chunk_id, "chunk_text": chunk_data["text"]} for chunk_id, chunk_data in doc_chunks.items()]
        db_file = db.query(DBFile).filter(DBFile.filename == filename).first()

        if not db_file:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found in the database")

        db_file.chunks = json.dumps(chunks_list)
        db.commit()

    await asyncio.sleep(2)
    return True


@app.post("/process_file/")
async def process_file(file_location: FileLocation, db: Session = Depends(get_db)):
    filename = file_location.filename
    file_record = None
    try:
        file_record = db.query(DBFile).filter(DBFile.filename == filename).first()

        if file_record:
            file_record.status = "Processing"
            db.commit()

            if await process_the_file(file_record.file_location, filename, db):
                file_record.status = "Done"
                db.commit()
                return {"status": "success", "message": "File processed successfully"}
            else:
                raise HTTPException(status_code=500, detail="Error processing file")
        else:
            raise HTTPException(status_code=404, detail="File not found")

    except Exception as e:
        if file_record:
            file_record.status = "Failed"
            db.commit()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/update_take_into_account/")
async def update_take_into_account(file_location: FileLocation, db: Session = Depends(get_db)):
    filename = file_location.filename
    try:
        file_record = db.query(DBFile).filter(DBFile.filename == filename).first()

        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")

        if file_record.take_into_account == "Enable":
            file_record.take_into_account = "Disable"
        else:
            file_record.take_into_account = "Enable"

        db.commit()

        return {"status": "success", "message": f"File take_into_account updated to {file_record.take_into_account}"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating take_into_account: {str(e)}")


@app.get("/get_uploaded_files/")
async def get_uploaded_files(db: Session = Depends(get_db)):
    files = db.query(DBFile).all()
    file_list = []
    for f_item in files:
        file_list.append({
            "filename": f_item.filename,
            "extension": f_item.extension,
            "upload_date": f_item.upload_date,
            "summary": f_item.summary,
            "status": f_item.status,
            "chunks": f_item.chunks,
            "take_into_account": f_item.take_into_account,
        })
    return file_list



@app.get("/get_file_chunks/{filename}")
async def get_file_chunks(filename: str, db: Session = Depends(get_db)):
    file_record = db.query(DBFile).filter(DBFile.filename == filename).first()

    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    chunks = json.loads(file_record.chunks) if file_record.chunks else []
    return {"filename": filename, "chunks": chunks}
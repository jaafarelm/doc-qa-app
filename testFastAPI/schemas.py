from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str


class AskRequest(BaseModel):
    question: str
    ##DocumentId: str

class AskResponse(BaseModel):
    answer: str

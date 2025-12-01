  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class InferenceItem(BaseModel):
    id: Optional[str] = Field(None, description="Client-provided id for this example")
    text: str = Field(..., description="Text input to the model")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional per-item metadata")


class InferenceRequest(BaseModel):
    items: List[InferenceItem]
    model_version: Optional[str] = Field(None, description="Requested model version / tag")


class Prediction(BaseModel):
    id: Optional[str]
    output: Dict[str, Any]


class InferenceResponse(BaseModel):
    flagged: bool = Field(False, description="Whether the input was flagged by safety")
    safety_reason: Optional[str] = Field(None)
    model_version: Optional[str] = Field(None)
    predictions: List[Prediction] = Field([], description="Model outputs per item")

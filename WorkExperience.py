from pydantic import BaseModel, Field

from typing import Optional

class WorkExperience(BaseModel):
    """Represents a single work experience entry."""
    job_title: str = Field(description="The job title held by the candidate.")
    company_name: str = Field(description="The name of the company.")
    start_date: Optional[str] = Field(description="The start date of the employment (e.g., YYYY-MM or Month YYYY).")
    end_date: Optional[str] = Field(description="The end date of the employment (e.g., YYYY-MM or Month YYYY), or 'Present'.")
    summary: str = Field(description="A brief summary of key responsibilities or achievements mentioned for this role.")
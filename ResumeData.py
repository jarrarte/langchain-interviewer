from WorkExperience import WorkExperience

from pydantic import BaseModel, Field

from typing import List
from typing import Optional


class ResumeData(BaseModel):
    """Represents structured data extracted from the resume."""
    candidate_name: Optional[str] = Field(default=None, description="The full name of the candidate identified in the resume.")
    experiences: List[WorkExperience] = Field(description="A list of work experience objects.")
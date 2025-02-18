from typing import Dict, Optional
from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field


class WriteFileInput(BaseModel):
    """Input for WriteFileTool."""

    file_path: str = Field(..., description="Path of the file to write to")
    text: str = Field(..., description="Text content to write to the file")


class CustomWriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Write text to a file"
    args_schema: type[BaseModel] = WriteFileInput

    def _run(self, file_path: str, text: str) -> str:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            raise ToolException(f"Error writing to file: {str(e)}")

    async def _arun(self, file_path: str, text: str) -> str:
        raise NotImplementedError("CustomWriteFileTool does not support async")

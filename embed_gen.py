from dotenv import load_dotenv
import os
import json
import hashlib
from pathlib import Path
from openai import AzureOpenAI
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from pathlib import Path
BASE_DIR = Path.cwd().parent
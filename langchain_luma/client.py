from .documents import DocumentsClient
from .http import HttpTransport
from .rag import RAGClient
from .sql import SQLClient
from .state import StateClient
from .stream import StreamClient
from .system import SystemClient
from .vectors import VectorsClient


class LumaClient:
    def __init__(
        self,
        url: str = "http://localhost:1234",
        api_key: str = "dev",
        timeout: int = 30,
        retries: int = 0,
    ):
        self._http = HttpTransport(base_url=url, api_key=api_key, timeout=timeout, retries=retries)

        self.system = SystemClient(self._http)
        self.state = StateClient(self._http)
        self.vectors = VectorsClient(self._http)
        self.documents = DocumentsClient(self._http)
        self.sql = SQLClient(self._http)
        self.rag = RAGClient(self._http)
        self.stream = StreamClient(self._http)

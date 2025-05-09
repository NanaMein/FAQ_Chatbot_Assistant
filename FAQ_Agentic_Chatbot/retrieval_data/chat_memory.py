from .rag_engine import lazy_loading, QueryEngineTool

lazily = lazy_loading()

obj = QueryEngineTool(lazily)


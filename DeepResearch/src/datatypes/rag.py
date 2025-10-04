"""
RAG (Retrieval-Augmented Generation) data types for DeepCritical research workflows.

This module defines Pydantic models for RAG components including embeddings,
vector stores, documents, and VLLM integration for local model hosting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, TYPE_CHECKING
from pydantic import BaseModel, Field, HttpUrl, model_validator

# Import existing dataclasses for alignment
from .chunk_dataclass import Chunk, generate_id
from .document_dataclass import Document as ChonkieDocument

if TYPE_CHECKING:
    import numpy as np


class SearchType(str, Enum):
    """Types of vector search operations."""

    SIMILARITY = "similarity"
    SEMANTIC = "semantic"
    MAX_MARGINAL_RELEVANCE = "mmr"
    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"
    HYBRID = "hybrid"  # Combines vector and keyword search


class EmbeddingModelType(str, Enum):
    """Types of embedding models supported by VLLM."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    CUSTOM = "custom"


class LLMModelType(str, Enum):
    """Types of LLM models supported by VLLM."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class VectorStoreType(str, Enum):
    """Types of vector stores supported."""

    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    ELASTICSEARCH = "elasticsearch"
    NEO4J = "neo4j"
    POSTGRES = "postgres"
    FAISS = "faiss"
    MILVUS = "milvus"


class Document(BaseModel):
    """Represents a document or record added to a vector store.

    Aligned with ChonkieDocument dataclass and enhanced for bioinformatics data.
    """

    id: str = Field(
        default_factory=lambda: generate_id("doc"),
        description="Unique document identifier",
    )
    content: str = Field(..., description="Document content/text")
    chunks: List[Chunk] = Field(default_factory=list, description="Document chunks")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    embedding: Optional[Union[List[float], "np.ndarray"]] = Field(
        None, description="Document embedding vector"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    # Bioinformatics-specific metadata fields
    bioinformatics_type: Optional[str] = Field(
        None, description="Type of bioinformatics data (GO, PubMed, GEO, etc.)"
    )
    source_database: Optional[str] = Field(
        None, description="Source database identifier"
    )
    cross_references: Dict[str, List[str]] = Field(
        default_factory=dict, description="Cross-references to other entities"
    )
    quality_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Quality score for the document"
    )

    def __len__(self) -> int:
        """Return the length of the document content."""
        return len(self.content)

    def __str__(self) -> str:
        """Return a string representation of the document."""
        return self.content

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the document."""
        self.chunks.append(chunk)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by its ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def to_chonkie_document(self) -> ChonkieDocument:
        """Convert to ChonkieDocument format."""
        return ChonkieDocument(
            id=self.id, content=self.content, chunks=self.chunks, metadata=self.metadata
        )

    @classmethod
    def from_chonkie_document(cls, doc: ChonkieDocument, **kwargs) -> "Document":
        """Create Document from ChonkieDocument."""
        return cls(
            id=doc.id,
            content=doc.content,
            chunks=doc.chunks,
            metadata=doc.metadata,
            **kwargs,
        )

    @classmethod
    def from_bioinformatics_data(cls, data: Any, **kwargs) -> "Document":
        """Create Document from bioinformatics data types."""
        from .bioinformatics import GOAnnotation, PubMedPaper, GEOSeries

        if isinstance(data, GOAnnotation):
            content = f"GO Annotation: {data.go_term.name}\nGene: {data.gene_symbol} ({data.gene_id})\nEvidence: {data.evidence_code.value}\nPaper: {data.title}\nAbstract: {data.abstract}"
            metadata = {
                "bioinformatics_type": "GO_annotation",
                "source_database": "GO",
                "pmid": data.pmid,
                "gene_id": data.gene_id,
                "gene_symbol": data.gene_symbol,
                "go_term_id": data.go_term.id,
                "evidence_code": data.evidence_code.value,
                "confidence_score": data.confidence_score,
            }
        elif isinstance(data, PubMedPaper):
            content = f"Title: {data.title}\nAbstract: {data.abstract}\nAuthors: {', '.join(data.authors)}\nJournal: {data.journal}"
            metadata = {
                "bioinformatics_type": "pubmed_paper",
                "source_database": "PubMed",
                "pmid": data.pmid,
                "doi": data.doi,
                "pmc_id": data.pmc_id,
                "journal": data.journal,
                "publication_date": data.publication_date.isoformat()
                if data.publication_date
                else None,
                "is_open_access": data.is_open_access,
                "mesh_terms": data.mesh_terms,
                "keywords": data.keywords,
            }
        elif isinstance(data, GEOSeries):
            content = f"GEO Series: {data.title}\nSummary: {data.summary}\nOrganism: {data.organism}\nDesign: {data.overall_design or 'N/A'}"
            metadata = {
                "bioinformatics_type": "geo_series",
                "source_database": "GEO",
                "series_id": data.series_id,
                "organism": data.organism,
                "platform_ids": data.platform_ids,
                "sample_ids": data.sample_ids,
                "pubmed_ids": data.pubmed_ids,
                "submission_date": data.submission_date.isoformat()
                if data.submission_date
                else None,
            }
        else:
            # Generic bioinformatics data
            content = str(data)
            metadata = {
                "bioinformatics_type": type(data).__name__.lower(),
                "source_database": "unknown",
            }

        return cls(
            content=content,
            metadata=metadata,
            bioinformatics_type=metadata.get("bioinformatics_type"),
            source_database=metadata.get("source_database"),
            **kwargs,
        )

    class Config:
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "id": "doc_001",
                "content": "This is a sample document about machine learning.",
                "chunks": [],
                "metadata": {
                    "source": "research_paper",
                    "author": "John Doe",
                    "year": 2024,
                    "bioinformatics_type": "pubmed_paper",
                    "source_database": "PubMed",
                },
                "bioinformatics_type": "pubmed_paper",
                "source_database": "PubMed",
            }
        }


class SearchResult(BaseModel):
    """Result from a vector search operation."""

    document: Document = Field(..., description="Retrieved document")
    score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Rank in search results")

    class Config:
        json_schema_extra = {
            "example": {
                "document": {
                    "id": "doc_001",
                    "content": "Sample content",
                    "metadata": {"source": "paper"},
                },
                "score": 0.95,
                "rank": 1,
            }
        }


class EmbeddingsConfig(BaseModel):
    """Configuration for embedding models."""

    model_type: EmbeddingModelType = Field(..., description="Type of embedding model")
    model_name: str = Field(..., description="Model name or identifier")
    api_key: Optional[str] = Field(None, description="API key for external services")
    base_url: Optional[HttpUrl] = Field(None, description="Base URL for API endpoints")
    num_dimensions: int = Field(
        1536, description="Number of dimensions in embedding vectors"
    )
    batch_size: int = Field(32, description="Batch size for embedding generation")
    max_retries: int = Field(3, description="Maximum retry attempts")
    timeout: float = Field(30.0, description="Request timeout in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "openai",
                "model_name": "text-embedding-3-small",
                "num_dimensions": 1536,
                "batch_size": 32,
            }
        }


class VLLMConfig(BaseModel):
    """Configuration for VLLM model hosting."""

    model_type: LLMModelType = Field(..., description="Type of LLM model")
    model_name: str = Field(..., description="Model name or path")
    host: str = Field("localhost", description="VLLM server host")
    port: int = Field(8000, description="VLLM server port")
    api_key: Optional[str] = Field(None, description="API key if required")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming responses")

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "huggingface",
                "model_name": "microsoft/DialoGPT-medium",
                "host": "localhost",
                "port": 8000,
                "max_tokens": 2048,
                "temperature": 0.7,
            }
        }


class VectorStoreConfig(BaseModel):
    """Configuration for vector store connections."""

    store_type: VectorStoreType = Field(..., description="Type of vector store")
    connection_string: Optional[str] = Field(
        None, description="Database connection string"
    )
    host: Optional[str] = Field(None, description="Vector store host")
    port: Optional[int] = Field(None, description="Vector store port")
    database: Optional[str] = Field(None, description="Database name")
    collection_name: Optional[str] = Field(None, description="Collection/index name")
    api_key: Optional[str] = Field(None, description="API key for cloud services")
    embedding_dimension: int = Field(1536, description="Embedding vector dimension")
    distance_metric: str = Field("cosine", description="Distance metric for similarity")
    index_type: Optional[str] = Field(None, description="Index type (e.g., HNSW, IVF)")

    class Config:
        json_schema_extra = {
            "example": {
                "store_type": "chroma",
                "host": "localhost",
                "port": 8000,
                "collection_name": "research_docs",
                "embedding_dimension": 1536,
            }
        }


class RAGQuery(BaseModel):
    """Query for RAG operations."""

    text: str = Field(..., description="Query text")
    search_type: SearchType = Field(
        SearchType.SIMILARITY, description="Type of search to perform"
    )
    top_k: int = Field(5, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(
        None, description="Minimum similarity score"
    )
    retrieval_query: Optional[str] = Field(
        None, description="Custom retrieval query for advanced stores"
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What is machine learning?",
                "search_type": "similarity",
                "top_k": 5,
                "filters": {"source": "research_paper"},
            }
        }


class RAGResponse(BaseModel):
    """Response from RAG operations."""

    query: str = Field(..., description="Original query")
    retrieved_documents: List[SearchResult] = Field(
        ..., description="Retrieved documents"
    )
    generated_answer: Optional[str] = Field(
        None, description="Generated answer from LLM"
    )
    context: str = Field(..., description="Context used for generation")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )
    processing_time: float = Field(..., description="Total processing time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "retrieved_documents": [],
                "generated_answer": "Machine learning is a subset of AI...",
                "context": "Based on the retrieved documents...",
                "processing_time": 1.5,
            }
        }


class RAGConfig(BaseModel):
    """Complete RAG system configuration."""

    embeddings: EmbeddingsConfig = Field(
        ..., description="Embedding model configuration"
    )
    llm: VLLMConfig = Field(..., description="LLM configuration")
    vector_store: VectorStoreConfig = Field(
        ..., description="Vector store configuration"
    )
    chunk_size: int = Field(1000, description="Document chunk size for processing")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    max_context_length: int = Field(4000, description="Maximum context length for LLM")
    enable_reranking: bool = Field(False, description="Enable document reranking")
    reranker_model: Optional[str] = Field(None, description="Reranker model name")

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values):
        """Validate RAG configuration."""
        embeddings = values.get("embeddings")
        vector_store = values.get("vector_store")

        if embeddings and vector_store:
            if embeddings.num_dimensions != vector_store.embedding_dimension:
                raise ValueError(
                    f"Embedding dimensions mismatch: "
                    f"embeddings.num_dimensions={embeddings.num_dimensions} "
                    f"!= vector_store.embedding_dimension={vector_store.embedding_dimension}"
                )

        return values

    class Config:
        json_schema_extra = {
            "example": {
                "embeddings": {
                    "model_type": "openai",
                    "model_name": "text-embedding-3-small",
                    "num_dimensions": 1536,
                },
                "llm": {
                    "model_type": "huggingface",
                    "model_name": "microsoft/DialoGPT-medium",
                    "host": "localhost",
                    "port": 8000,
                },
                "vector_store": {"store_type": "chroma", "embedding_dimension": 1536},
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        }


# Abstract base classes for implementations


class Embeddings(ABC):
    """Abstract base class for embedding generation."""

    def __init__(self, config: EmbeddingsConfig):
        self.config = config

    @property
    def num_dimensions(self) -> int:
        """The number of dimensions in the resulting vector."""
        return self.config.num_dimensions

    @abstractmethod
    async def vectorize_documents(
        self, document_chunks: List[str]
    ) -> List[List[float]]:
        """Generate document embeddings for a list of chunks."""
        pass

    @abstractmethod
    async def vectorize_query(self, text: str) -> List[float]:
        """Generate embeddings for the query string."""
        pass

    @abstractmethod
    def vectorize_documents_sync(self, document_chunks: List[str]) -> List[List[float]]:
        """Synchronous version of vectorize_documents()."""
        pass

    @abstractmethod
    def vectorize_query_sync(self, text: str) -> List[float]:
        """Synchronous version of vectorize_query()."""
        pass


class VectorStore(ABC):
    """Abstract base class for vector store implementation."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        self.config = config
        self.embeddings = embeddings

    @abstractmethod
    async def add_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add a list of documents to the vector store and return their unique identifiers."""
        pass

    @abstractmethod
    async def add_document_chunks(
        self, chunks: List[Chunk], **kwargs: Any
    ) -> List[str]:
        """Add document chunks to the vector store."""
        pass

    @abstractmethod
    async def add_document_text_chunks(
        self, document_texts: List[str], **kwargs: Any
    ) -> List[str]:
        """Add document text chunks to the vector store (legacy method)."""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete the specified list of documents by their record identifiers."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        search_type: SearchType,
        retrieval_query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Search for documents using text query."""
        pass

    @abstractmethod
    async def search_with_embeddings(
        self,
        query_embedding: List[float],
        search_type: SearchType,
        retrieval_query: Optional[str] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Search for documents using embedding vector."""
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        pass

    @abstractmethod
    async def update_document(self, document: Document) -> bool:
        """Update an existing document."""
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: VLLMConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self, prompt: str, context: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Generate text using the LLM."""
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, context: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using the LLM."""
        pass


class RAGSystem(BaseModel):
    """Complete RAG system implementation."""

    config: RAGConfig = Field(..., description="RAG system configuration")
    embeddings: Optional[Embeddings] = Field(None, description="Embeddings provider")
    vector_store: Optional[VectorStore] = Field(None, description="Vector store")
    llm: Optional[LLMProvider] = Field(None, description="LLM provider")

    async def initialize(self) -> None:
        """Initialize the RAG system components."""
        # This would be implemented by concrete classes
        pass

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        return await self.vector_store.add_documents(documents)

    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """Perform a complete RAG query."""
        import time

        start_time = time.time()

        if not self.vector_store or not self.llm:
            raise RuntimeError("RAG system not fully initialized")

        # Retrieve relevant documents
        search_results = await self.vector_store.search(
            query=rag_query.text,
            search_type=rag_query.search_type,
            retrieval_query=rag_query.retrieval_query,
            top_k=rag_query.top_k,
            score_threshold=rag_query.score_threshold,
            filters=rag_query.filters,
        )

        # Build context from retrieved documents
        context_parts = []
        for result in search_results:
            context_parts.append(f"Document {result.rank}: {result.document.content}")

        context = "\n\n".join(context_parts)

        # Generate answer using LLM
        prompt = f"""Based on the following context, please answer the question: {rag_query.text}

Context:
{context}

Answer:"""

        generated_answer = await self.llm.generate(prompt, context=context)

        processing_time = time.time() - start_time

        return RAGResponse(
            query=rag_query.text,
            retrieved_documents=search_results,
            generated_answer=generated_answer,
            context=context,
            processing_time=processing_time,
        )

    class Config:
        arbitrary_types_allowed = True


class BioinformaticsRAGSystem(RAGSystem):
    """Specialized RAG system for bioinformatics data fusion and reasoning."""

    def __init__(self, config: RAGConfig, **kwargs):
        super().__init__(config=config, **kwargs)
        self.bioinformatics_data_cache: Dict[str, Any] = {}

    async def add_bioinformatics_data(self, data: List[Any]) -> List[str]:
        """Add bioinformatics data to the vector store."""
        documents = []
        for item in data:
            doc = Document.from_bioinformatics_data(item)
            documents.append(doc)

        return await self.add_documents(documents)

    async def query_bioinformatics(
        self, query: BioinformaticsRAGQuery
    ) -> BioinformaticsRAGResponse:
        """Perform a specialized bioinformatics RAG query."""
        import time

        start_time = time.time()

        if not self.vector_store or not self.llm:
            raise RuntimeError("RAG system not fully initialized")

        # Build enhanced filters for bioinformatics data
        enhanced_filters = query.filters or {}
        if query.bioinformatics_types:
            enhanced_filters["bioinformatics_type"] = {
                "$in": query.bioinformatics_types
            }
        if query.source_databases:
            enhanced_filters["source_database"] = {"$in": query.source_databases}
        if query.evidence_codes:
            enhanced_filters["evidence_code"] = {"$in": query.evidence_codes}
        if query.organisms:
            enhanced_filters["organism"] = {"$in": query.organisms}
        if query.gene_symbols:
            enhanced_filters["gene_symbol"] = {"$in": query.gene_symbols}
        if query.quality_threshold:
            enhanced_filters["quality_score"] = {"$gte": query.quality_threshold}

        # Retrieve relevant documents with bioinformatics filters
        search_results = await self.vector_store.search(
            query=query.text,
            search_type=query.search_type,
            retrieval_query=query.retrieval_query,
            top_k=query.top_k,
            score_threshold=query.score_threshold,
            filters=enhanced_filters,
        )

        # Build context from retrieved documents
        context_parts = []
        bioinformatics_summary = {
            "total_documents": len(search_results),
            "bioinformatics_types": set(),
            "source_databases": set(),
            "evidence_codes": set(),
            "organisms": set(),
            "gene_symbols": set(),
        }

        cross_references = {}

        for result in search_results:
            doc = result.document
            context_parts.append(f"Document {result.rank}: {doc.content}")

            # Extract bioinformatics metadata
            if doc.bioinformatics_type:
                bioinformatics_summary["bioinformatics_types"].add(
                    doc.bioinformatics_type
                )
            if doc.source_database:
                bioinformatics_summary["source_databases"].add(doc.source_database)

            # Extract metadata for summary
            metadata = doc.metadata
            if "evidence_code" in metadata:
                bioinformatics_summary["evidence_codes"].add(metadata["evidence_code"])
            if "organism" in metadata:
                bioinformatics_summary["organisms"].add(metadata["organism"])
            if "gene_symbol" in metadata:
                bioinformatics_summary["gene_symbols"].add(metadata["gene_symbol"])

            # Collect cross-references
            if doc.cross_references:
                for ref_type, refs in doc.cross_references.items():
                    if ref_type not in cross_references:
                        cross_references[ref_type] = set()
                    cross_references[ref_type].update(refs)

        # Convert sets to lists for JSON serialization
        for key in bioinformatics_summary:
            if isinstance(bioinformatics_summary[key], set):
                bioinformatics_summary[key] = list(bioinformatics_summary[key])

        for key in cross_references:
            cross_references[key] = list(cross_references[key])

        context = "\n\n".join(context_parts)

        # Generate specialized prompt for bioinformatics
        prompt = f"""Based on the following bioinformatics data, please provide a comprehensive answer to: {query.text}

Context from bioinformatics databases:
{context}

Please provide:
1. A direct answer to the question
2. Key findings from the data
3. Relevant gene symbols, GO terms, or other identifiers mentioned
4. Confidence level based on the evidence quality

Answer:"""

        generated_answer = await self.llm.generate(prompt, context=context)

        processing_time = time.time() - start_time

        # Calculate quality metrics
        quality_metrics = {
            "average_score": sum(r.score for r in search_results) / len(search_results)
            if search_results
            else 0.0,
            "high_quality_docs": sum(1 for r in search_results if r.score > 0.8),
            "evidence_diversity": len(bioinformatics_summary["evidence_codes"]),
            "source_diversity": len(bioinformatics_summary["source_databases"]),
        }

        return BioinformaticsRAGResponse(
            query=query.text,
            retrieved_documents=search_results,
            generated_answer=generated_answer,
            context=context,
            processing_time=processing_time,
            bioinformatics_summary=bioinformatics_summary,
            cross_references=cross_references,
            quality_metrics=quality_metrics,
        )

    async def fuse_bioinformatics_data(
        self, data_sources: Dict[str, List[Any]]
    ) -> List[Document]:
        """Fuse multiple bioinformatics data sources into unified documents."""
        fused_documents = []

        for source_name, data_list in data_sources.items():
            for item in data_list:
                doc = Document.from_bioinformatics_data(item)
                doc.metadata["fusion_source"] = source_name
                fused_documents.append(doc)

        # Add cross-references between related documents
        self._add_cross_references(fused_documents)

        return fused_documents

    def _add_cross_references(self, documents: List[Document]) -> None:
        """Add cross-references between related documents."""
        # Group documents by common identifiers
        gene_groups = {}
        pmid_groups = {}

        for doc in documents:
            metadata = doc.metadata

            # Group by gene symbols
            if "gene_symbol" in metadata:
                gene_symbol = metadata["gene_symbol"]
                if gene_symbol not in gene_groups:
                    gene_groups[gene_symbol] = []
                gene_groups[gene_symbol].append(doc.id)

            # Group by PMIDs
            if "pmid" in metadata:
                pmid = metadata["pmid"]
                if pmid not in pmid_groups:
                    pmid_groups[pmid] = []
                pmid_groups[pmid].append(doc.id)

        # Add cross-references to documents
        for doc in documents:
            metadata = doc.metadata
            cross_refs = {}

            if "gene_symbol" in metadata:
                gene_symbol = metadata["gene_symbol"]
                related_docs = [
                    doc_id for doc_id in gene_groups[gene_symbol] if doc_id != doc.id
                ]
                if related_docs:
                    cross_refs["related_gene_docs"] = related_docs

            if "pmid" in metadata:
                pmid = metadata["pmid"]
                related_docs = [
                    doc_id for doc_id in pmid_groups[pmid] if doc_id != doc.id
                ]
                if related_docs:
                    cross_refs["related_pmid_docs"] = related_docs

            if cross_refs:
                doc.cross_references = cross_refs


class BioinformaticsRAGQuery(BaseModel):
    """Specialized RAG query for bioinformatics data."""

    text: str = Field(..., description="Query text")
    search_type: SearchType = Field(
        SearchType.SIMILARITY, description="Type of search to perform"
    )
    top_k: int = Field(5, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(
        None, description="Minimum similarity score"
    )
    retrieval_query: Optional[str] = Field(
        None, description="Custom retrieval query for advanced stores"
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

    # Bioinformatics-specific filters
    bioinformatics_types: Optional[List[str]] = Field(
        None, description="Filter by bioinformatics data types"
    )
    source_databases: Optional[List[str]] = Field(
        None, description="Filter by source databases"
    )
    evidence_codes: Optional[List[str]] = Field(
        None, description="Filter by GO evidence codes"
    )
    organisms: Optional[List[str]] = Field(None, description="Filter by organisms")
    gene_symbols: Optional[List[str]] = Field(
        None, description="Filter by gene symbols"
    )
    quality_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum quality score"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What genes are involved in DNA damage response?",
                "search_type": "similarity",
                "top_k": 10,
                "bioinformatics_types": ["GO_annotation", "pubmed_paper"],
                "source_databases": ["GO", "PubMed"],
                "evidence_codes": ["IDA", "EXP"],
                "quality_threshold": 0.8,
            }
        }


class BioinformaticsRAGResponse(BaseModel):
    """Enhanced RAG response for bioinformatics data."""

    query: str = Field(..., description="Original query")
    retrieved_documents: List[SearchResult] = Field(
        ..., description="Retrieved documents"
    )
    generated_answer: Optional[str] = Field(
        None, description="Generated answer from LLM"
    )
    context: str = Field(..., description="Context used for generation")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )
    processing_time: float = Field(..., description="Total processing time in seconds")

    # Bioinformatics-specific response data
    bioinformatics_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of bioinformatics data"
    )
    cross_references: Dict[str, List[str]] = Field(
        default_factory=dict, description="Cross-references found"
    )
    quality_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Quality metrics for retrieved data"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What genes are involved in DNA damage response?",
                "retrieved_documents": [],
                "generated_answer": "Based on the retrieved GO annotations and PubMed papers...",
                "context": "Context from retrieved documents...",
                "processing_time": 2.1,
                "bioinformatics_summary": {
                    "total_annotations": 15,
                    "unique_genes": 8,
                    "evidence_types": ["IDA", "EXP", "IPI"],
                },
            }
        }


class RAGWorkflowState(BaseModel):
    """State for RAG workflow execution."""

    query: str = Field(..., description="Original query")
    rag_config: RAGConfig = Field(..., description="RAG system configuration")
    documents: List[Document] = Field(
        default_factory=list, description="Documents to process"
    )
    chunks: List[Chunk] = Field(default_factory=list, description="Document chunks")
    rag_response: Optional[RAGResponse] = Field(None, description="RAG response")
    bioinformatics_response: Optional[BioinformaticsRAGResponse] = Field(
        None, description="Bioinformatics RAG response"
    )
    processing_steps: List[str] = Field(
        default_factory=list, description="Processing steps completed"
    )
    errors: List[str] = Field(
        default_factory=list, description="Any errors encountered"
    )

    # Bioinformatics-specific state
    bioinformatics_data: Dict[str, Any] = Field(
        default_factory=dict, description="Bioinformatics data being processed"
    )
    fusion_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Data fusion metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "rag_config": {},
                "documents": [],
                "chunks": [],
                "processing_steps": ["initialized", "documents_loaded"],
                "bioinformatics_data": {"go_annotations": [], "pubmed_papers": []},
            }
        }

# RAG Techniques Comparison Template

Use this document for your course deliverable comparing retrieval strategies.

## Baseline (implemented)
- Chunking: fixed-size character chunks with overlap
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval: Chroma similarity top-k

## Additional techniques to compare
1. BM25 lexical retrieval
2. Hybrid retrieval (BM25 + vector)
3. Multi-query retrieval (query rewriting)
4. Parent-document retrieval
5. Reranking with cross-encoder

## Evaluation protocol
1. Build a fixed evaluation question set (15-30 questions).
2. Run each technique with the same corpus and model.
3. Record:
   - Retrieved document IDs/chunks
   - Response latency (retrieval + generation)
   - Citation relevance
   - Answer quality (human judgment rubric)

## Suggested result table

| Technique | Avg Retrieval Time (ms) | Avg Total Time (s) | Retrieval Quality (1-5) | Answer Quality (1-5) | Notes |
|---|---:|---:|---:|---:|---|
| Baseline similarity |  |  |  |  |  |
| BM25 |  |  |  |  |  |
| Hybrid |  |  |  |  |  |
| Multi-query |  |  |  |  |  |
| Reranked |  |  |  |  |  |

## Interpretation prompts
- Which techniques changed retrieved chunks most significantly?
- Which techniques improved grounding/citation reliability?
- Why did some methods increase latency?
- Which approach gives best quality/latency tradeoff for your deployment constraints?

ShopTalk Project Charter

Status: Draft v1 · Date: 2026-10-24 · Repo: github.com/anbuaru/shoptalk

1. Problem

Keyword search handles shopping queries poorly. A query like "show me a women’s black sandals under $30" mixes a product type, a color, a gender, and a price limit, and a keyword match on those words returns irrelevant or incomplete results. It also misses synonyms ("tee" vs "shirt") and vague requests ("a gift for a coffee lover").

ShopTalk is a shopping assistant that understands natural-language queries, retrieves the most relevant products using embeddings and a vector database, and answers in natural language with product identifiers. It uses Retrieval-Augmented Generation (RAG) with open-source models. The app accepts text first, and voice input is added after the text version works.

2. Goals
Return relevant products for natural-language queries, including attribute constraints (color, price, gender, brand).
Generate a short, grounded recommendation that only mentions retrieved products and always lists their IDs.
Show that fine-tuning the embedding model on our data improves retrieval over pretrained models.
Respond fast enough to feel interactive.
Accept spoken queries as well as typed ones (voice is mandatory, built after text).
Deliver a deployed, documented prototype with a REST endpoint and a basic UI.
3. Example queries
#	Query	Type
1	“Women’s sandals under $30”	Attribute and price filter
2	“Waterproof hiking boots”	Attribute
3	“A black leather wallet”	Spoken (voice phase)
4	“A gift for someone who loves coffee”	Vague / semantic
5	“A comfortable chair for a small living room”	Vague / semantic
6	“Gold earrings under $25”	Attribute and price filter
7	“Show me something similar but cheaper”	Follow-up

These are initial targets. Revisit them after the first baseline results (Phase 5) and record any changes with the reason.

Area	Metric	Target
Retrieval	Precision@5 on the held-out test set	≥ 0.70
Retrieval	Fine-tuned model vs best pretrained baseline	Fine-tuned is better on Precision@5
Generation	Answers that mention only retrieved products (manual review of 30-50 answers)	≥ 90%
Generation	Answers that list product IDs	100%
Latency	End-to-end P95 (text query, on the deployment hardware)	≤ 5 s
Latency	End-to-end P99	Reported (no hard target)
Voice	Word error rate on 10-20 spoken test queries	Reported, with a target set after the first test
Voice	Added latency from speech-to-text	Reported separately
5. Scope

In scope

Preprocessing of the ABO dataset (text, metadata, image captions)
Embedding generation with pretrained and fine-tuned models (triplet-style loss, optional LoRA/QLoRA)
Chroma vector database (Milvus or FAISS as a comparison)
Image captioning to enrich product text
Open-source LLM for answer generation, with a latency comparison
RAG pipeline exposed as a REST endpoint (FastAPI)
Gradio or Streamlit UI with product IDs/links and thumbs up/down feedback
Text input first, then voice input (Phase 10)
Evaluation: retrieval precision, qualitative generation review, P95/P99 latency
Docker, AWS deployment, and documentation

Out of scope for now (revisit only if time allows)

Image-based search
Personalization from user history
Fine-tuning the generation LLM
Multilingual support

Optional

Follow-up queries with conversation history
Feedback-driven improvements beyond logging
6. Dataset

Amazon Berkeley Objects (ABO). Fields used: product title, description, bullet points, keywords, brand, product type, color, price (where available), and the main product image.

7. Risks and mitigations
Risk	Mitigation
Price data is sparse in ABO, which weakens "under $50" queries	Check price coverage in the first data exploration. If it is too low, document it and enrich the data or narrow the categories
Colab free-tier GPU limits slow captioning and fine-tuning	Run captioning as a batch job, save outputs to Drive, use Kaggle as a backup, and restrict to a few product categories
LLM generation latency exceeds the P95 target	Use small quantized models, limit retrieved context to the top 3-5 products, and compare several models
The model invents products or details	Prompt it to recommend only from retrieved products, and measure groundedness in review
Speech recognition mishears product names, brands, and sizes	Evaluate transcription and retrieval separately, show the transcribed query in the UI, and compare ASR model sizes
AWS costs run over	Deploy late, use the smallest instance that works, and shut it down after the demo
8. Plan

Nine weeks, by phase (details in the project plan):

Weeks	Phase
1	Setup and research (Phases 1-2)
2-3	Data preparation and evaluation set (Phases 3-4)
4	Embedding experiments and fine-tuning (Phase 5)
5	RAG pipeline and API (Phases 6-7)
6	UI (Phase 8)
7	Testing and evaluation of the text version (Phase 9)
8	Voice input (Phase 10)
9	Deployment and documentation (Phases 11-12)
9. Deliverables
Colab notebooks and fine-tuned models
Preprocessed data with documentation of the steps
RAG and inference code, plus the AWS deployment code and documentation
A working bot with UI, and a video recording of it
requirements.txt, Dockerfile, README, and deployment steps
A system architecture and implementation document
Initial results, experiences, and a plan for further testing and improvement

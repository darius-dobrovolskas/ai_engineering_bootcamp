# AI Project Canvas – Amazon Inventory Shopping Assistant

**Prepared by:** Darius  
**Prepared for:** _Aurimas Griciūnas_  
**Date:** 2025-07-01  
**Version:** _v1.0_

---

## 📌 Problem Statement

Aurimas Griciūnas owns warehouses and manages inventory data but lacks automation in his current processes. Customers and product data are not integrated into any intelligent system. <br>
The **goal** is to build an AI-powered assistant that helps users explore and interact with this inventory by answering product-related questions and recommending shopping actions. <br>Traditional manual methods don’t scale well with millions of products and reviews, hence the need for a generative AI-based assistant that improves over time.

---

## 📚 Data & Knowledge

- **Source**: Publicly available Amazon inventory and review datasets (filtered from 15M products to a subset of ~17,000 electronics items).
- **Filtering**:
  - Products launched in 2022 or 2023 only.
  - Only items with 100+ reviews were selected.
  - A random sample of 1,000 items is initially used for development (with ~100,000 total reviews).
- **Format**: JSON files.
- **Future Data**: Stock and warehouse data will be added later for more complex use cases (logistics, availability, etc.).

---

## 📈 Performance Metrics & Evaluation Rules

- **Strategic (North Star Metric)**:
    - Monthly revenue (EUR/month)

- **Technical Metrics**
    - Token/session/transaction cost (EUR)
    - Time to first token (TTFT)
    - Latency (response time)
    - Accuracy of assistant responses
    - User feedback scores per interaction

- **Operational Metrics**
    - Time from initial query to completed purchase
    - Cart abandonment rate
    - Conversion rate (conversations to purchases)
    - Cost per transaction

- **User/Business Metrics**
    - Repeat purchase rate
    - Overall user satisfaction (e.g. NPS)
---

## 👥 Resources & Stakeholders

- **Stakeholders**: Aurimas Griciūnas
- **AI Engineers / Developers**: Darius
- **Infrastructure**: Streamlit app frontend, <br>vector database (e.g., Qdrant), <br>LLM APIs.


---

## 🔌 Deployment & Integration

- **Phase 1**: Standalone Streamlit app with chatbot interface
- **Phase 2**: Integration of product database, reviews, and feedback loo
- **Phase 3**: Multi-agent architecture for booking and stock management
---

## 🤖 AI Approach & Methodology

- **Techniques**: Retrieval-Augmented Generation (RAG)
- **LLMs**: Start with zero/few-shot prompting
- **Prompting**: Dynamic template prompts based on item attributes + review context
- **Knowledge Retrieval**: Embedding-based search from vector database
- **Agentic Systems**: Multi-agent workflow to manage queries, bookings, and cart creation

---

## ⚠️ Risks & Mitigation

- Scalability: Difficulty processing full dataset <br>
→ Use a modular pipeline design, start with a 1,000-item subset, and implement batch processing and vector DB indexing strategies.

- User experience: Confusing or irrelevant suggestions <br>
→ Implement continuous user feedback collection and active learning mechanisms to improve prompt targeting and ranking logic.

- Quality control: Inaccurate, hallucinated, or misleading LLM responses <br>
→ Integrate LLM guardrails (e.g., content moderation, truth-checking modules), define strict prompt scopes, and verify outputs via fallback logic or human-in-the-loop review.

- Model drift or prompt degradation: Performance may decline over time or across product domains <br>
→ Establish regular evaluations, prompt audits, and schedule fine-tuning cycles with collected domain data.

---

## 🗓️ Timeline & Milestones

| Week | Milestone                                      |
|------|------------------------------------------------|
| 1    | Data sampling and Streamlit app setup          |
| 2    | Implement basic RAG-based Q&A                  |
| 3    | Enable filtering and item-level search         |
| 4    | Add feedback UI and start gathering signals    |
| 5    | Introduce synthetic stock data and warehouse logic |
| 6    | Add multi-agent capability (e.g., cart builder)|
| 7    | Test end-to-end flow, deploy MVP               |
| 8    | Iterate on feedback and prepare final demo     |
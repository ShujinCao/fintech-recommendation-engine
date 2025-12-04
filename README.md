# fintech-recommendation-engine
This repository implements an end-to-end **FinTech product recommendation system** that can scale to millions of users.  - **Two-Tower Retrieval Model** for candidate generation - **Gradient-Boosted Tree Ranker** for final scoring

### End-to-End Training Pipeline
```bash scripts/run_full_pipeline.sh```
### Start the Recommendation API
```bash scripts/start_server.sh```

### Once the server is running, open:
Swagger UI (interactive API docs):
http://127.0.0.1:8000/docs

Example Recommendation Endpoint:
http://127.0.0.1:8000/recommend/42?k=10

The system demonstrates:

- **Data Pipeline Architecture** (batch + near-real-time)
- **Feature Engineering & Preprocessing**
- **Two-Tower Retrieval Model** for candidate generation
- **Gradient-Boosted Tree Ranker** for final scoring
- **Model Training & Offline Evaluation**
- **Real-Time Inference API** with FastAPI
- **Monitoring & Feedback Loop Design**
- **A/B Testing & Continuous Model Lifecycle Management**

All data in this repo is **synthetic** and generated via scripts under `src/data`.

## High-Level Architecture

1. **Synthetic Data Generation**
   - Users: demographics, income, risk tolerance, product holdings.
   - Products: type (loan, savings, insurance), risk/return, eligibility.
   - Interactions: views, clicks, applications, approvals.

2. **Feature Engineering**
   - User features: profile, aggregated behavior, risk profile.
   - Product features: category, risk, pricing, historical demand.
   - Interaction features: recency, frequency, product–user compatibility.

3. **Modeling**
   - **Two-Tower Model (PyTorch)** for user/product embeddings and candidate retrieval.
   - **Ranker (LightGBM)** using:
     - Two-Tower similarity scores
     - User/product features
     - Interaction & context features

4. **Serving**
   - **FastAPI** endpoint:
     - `GET /recommend/{user_id}?k=10`
   - Loads pre-trained towers, ranker, and feature encoders.

5. **Monitoring & Feedback**
   - Logging of served recommendations and user responses as synthetic events.
   - Placeholder hooks for:
     - CTR/Conversion tracking
     - Drift detection
     - A/B testing cohorts

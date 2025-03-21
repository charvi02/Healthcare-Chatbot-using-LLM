The Context-Aware Medical Assistant is an AI-powered tool designed to aid doctors in diagnosing diseases using Large Language Models (LLMs). It enhances real-time diagnosis accuracy while ensuring human oversight.  
1. Data Handling: Medical textbooks  were vectorized using **PyPDF2** and **ChromaDB**, with embeddings created via nomic-embed-text.  
2. Model Implementation: Llama 3.1 and BioMistral were deployed using **Ollama** with a **RAG pipeline** via LangChain. **Fine-tuning** was performed using Unsloth++.  
3. Evaluation:Doctors reviewed model predictions on diagnosis, medication, and severity through Google Forms, favoring **Model 3** (finetuned Llama 3.1 with RAG).  
4. Impact: Reduces diagnosis time, improves healthcare access, and supports **doctors in underserved areas**. It helps medical professionals make faster, informed decisions while improving patient care efficiency.

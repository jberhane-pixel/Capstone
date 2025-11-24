import json
from sentence_transformers import SentenceTransformer, util

#Load Model

model = SentenceTransformer('all-MiniLM-L6-v2')

#Load Scraped Jobs from JSON

def load_jobs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    return jobs

#Compute Similarity Scores (cosine similarity)

def compute_similarity(resume_text, job_descriptions):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
    
    cosine_scores = util.pytorch_cos_sim(resume_embedding, job_embeddings).item()

    return cosine_scores

#Rank Jobs Based on Similarity Scores

def rank_jobs(text_input, jobs):
    results = []

    for job in jobs:
        job_desc = job.get('description_text', '')
        score = compute_similarity(text_input, job_desc)
        results.append({
            'title': job.get('title', 'N/A'),
            'company': job.get('compa ny', 'N/A'),
            'score': round(score *100, 2),
            'description': job_desc,
            'url': job.get('source_url', 'N/A')
        })

    ranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return ranked_results

# Example Usage
if __name__ == "__main__":
    jobs = load_jobs('scraped_jobs.json')

    resume = """
    Experienced software engineer with a strong background in Python, machine learning, and data analysis. Looking for job in NVidia or similar companies."""
    
    matches = rank_jobs(resume, jobs)
    for match in matches[:5]:  # Top 5 matches
        print(f"Title: {match['title']}, Company: {match['company']}, Score: {match['score']}%")
        print(f"URL: {match['url']}")
        print(f"Description: {match['description'][:200]}...")  # Print first 200 chars
        print("-" * 80)

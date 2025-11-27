import json
import argparse
from typing import List, Dict, Any, Set, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class UnsupervisedJobRecommender:
    """
    Unsupervised job recommender using:
      - TF-IDF embeddings
      - K-Means clustering
      - User feedback to penalize disliked clusters
    """

    def __init__(self, n_clusters: int = 3, max_features: int = 5000):
        self.n_clusters = n_clusters
        self.max_features = max_features

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kmeans: Optional[KMeans] = None

        self.jobs: List[Dict[str, Any]] = []
        self.job_texts: List[str] = []

        self.job_embeddings = None  # TF-IDF matrix
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_penalties: Optional[np.ndarray] = None

        # feedback
        self.disliked_job_indices: Set[int] = set()
        self.disliked_clusters: Set[int] = set()

    # -------------------------------------------------------
    # Load job data
    # -------------------------------------------------------
    def load_jobs(self, json_path: str) -> None:
        print(f"Loading jobs from: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.jobs = json.load(f)

        # Try multiple possible description keys just in case
        self.job_texts = []
        for job in self.jobs:
            desc = (
                job.get("description_text")
                or job.get("description")
                or job.get("Desc")
                or job.get("desc")
                or ""
            )
            self.job_texts.append(desc)

        if not self.job_texts or all(not t.strip() for t in self.job_texts):
            raise ValueError("No job descriptions found in JSON file.")

        print(f"Loaded {len(self.jobs)} jobs.\n")

    # -------------------------------------------------------
    # Fit TF-IDF + KMeans
    # -------------------------------------------------------
    def fit(self) -> None:
        if not self.job_texts:
            raise ValueError("No jobs loaded. Call load_jobs() first.")

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features
        )
        self.job_embeddings = self.vectorizer.fit_transform(self.job_texts)

        # KMeans clustering
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10  # keep compatible with most sklearn versions
        )
        self.cluster_labels = self.kmeans.fit_predict(self.job_embeddings)

        # penalties for each cluster
        self.cluster_penalties = np.zeros(self.n_clusters, dtype=float)

        print("Model fitted: TF-IDF + KMeans complete.\n")

    # -------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------
    def _embed_text(self, text: str):
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        return self.vectorizer.transform([text])

    def _base_similarity_scores(self, resume_text: str) -> np.ndarray:
        resume_emb = self._embed_text(resume_text)
        scores = cosine_similarity(resume_emb, self.job_embeddings)[0]
        return scores

    def _apply_feedback(self, scores: np.ndarray) -> np.ndarray:
        if self.cluster_labels is None or self.cluster_penalties is None:
            return scores

        adjusted = scores.copy()
        for idx, cluster in enumerate(self.cluster_labels):
            # subtract cluster penalty
            adjusted[idx] -= self.cluster_penalties[cluster]

            # crush explicitly disliked jobs
            if idx in self.disliked_job_indices:
                adjusted[idx] -= 1.0

        return adjusted

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------
    def recommend(self, resume_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        base_scores = self._base_similarity_scores(resume_text)
        adjusted_scores = self._apply_feedback(base_scores)

        top_indices = np.argsort(-adjusted_scores)[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            job = self.jobs[idx]
            results.append({
                "index": idx,
                "cluster": int(self.cluster_labels[idx]) if self.cluster_labels is not None else -1,
                "title": job.get("title", "N/A"),
                "company": job.get("company", job.get("Company", "N/A")),
                "location": job.get("location", job.get("Location", "N/A")),
                "score": float(adjusted_scores[idx]),
                "url": job.get("url", job.get("URL", "N/A"))
            })
        return results

    def register_dislikes(self, disliked_indices: List[int], penalty: float = 0.15) -> None:
        if self.cluster_labels is None or self.cluster_penalties is None:
            return

        for idx in disliked_indices:
            if idx < 0 or idx >= len(self.jobs):
                continue

            self.disliked_job_indices.add(idx)
            cluster = int(self.cluster_labels[idx])
            self.disliked_clusters.add(cluster)
            self.cluster_penalties[cluster] += penalty

        print("Feedback registered: similar jobs will be ranked lower.\n")

    # -------------------------------------------------------
    # Pretty print
    # -------------------------------------------------------
    @staticmethod
    def print_recommendations(recs: List[Dict[str, Any]]) -> None:
        if not recs:
            print("No recommendations to show.")
            return

        print("\nTop Recommendations:")
        print("-" * 60)
        for i, job in enumerate(recs, start=1):
            print(f"[{i}] internal_index={job['index']} | cluster={job['cluster']}")
            print(f"     Title   : {job['title']}")
            print(f"     Company : {job['company']}")
            print(f"     Location: {job['location']}")
            print(f"     Score   : {job['score']:.4f}")
            print(f"     URL     : {job['url']}")
            print("-" * 60)


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unsupervised job matcher with user feedback.")

    # Default JSON path for your Capstone repo
    parser.add_argument(
        "--jobs",
        type=str,
        default="scraped_jobs.json",   # your real scraped file in this folder
        help="Path to scraped job JSON file."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Number of KMeans clusters."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many jobs to show per query."
    )

    args = parser.parse_args()

    rec = UnsupervisedJobRecommender(n_clusters=args.clusters)
    rec.load_jobs(args.jobs)
    rec.fit()

    print("Unsupervised Job Matcher (TF-IDF + KMeans, with feedback)")
    print("Type 'q' at any prompt to quit.\n")

    while True:
        resume_text = input("Paste your resume summary or skills:\n> ").strip()
        if resume_text.lower() == "q":
            break
        if not resume_text:
            print("Please type something.\n")
            continue

        results = rec.recommend(resume_text, top_k=args.top_k)
        rec.print_recommendations(results)

        feedback = input(
            "\nDislike any of these jobs? Enter numbers (e.g., 1,3) "
            "or press Enter to accept, or 'q' to quit:\n> "
        ).strip()

        if feedback.lower() == "q":
            break

        if feedback:
            try:
                nums = [int(x.strip()) for x in feedback.split(",") if x.strip()]
                disliked = [
                    results[n - 1]["index"]
                    for n in nums
                    if 1 <= n <= len(results)
                ]
                rec.register_dislikes(disliked)
            except Exception:
                print("Invalid format. Use: 1,3")

        print("\n--- Updated recommendations next cycle ---\n")


if __name__ == "__main__":
    main()

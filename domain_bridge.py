"""
domain_bridge.py
================
Solves the vocabulary mismatch problem between real-world CV language
and the dataset's specific vocabulary.

When someone uploads a CV saying "Python, Machine Learning, Deep Learning"
the model needs to understand this maps to ML Engineer / Software Engineer
not Nurse or Pharmacist.

Two mechanisms:
1. REAL_WORLD_BRIDGE: maps common real-world terms to dataset vocabulary terms
2. DOMAIN_DETECTOR: detects the user's domain from any text/keywords
3. DOMAIN_CAREERS: maps domains to valid career predictions
4. Career reranking: within the predicted careers, rerank based on detected domain
"""

# REAL-WORLD → DATASET VOCABULARY BRIDGE 
# Keys = what people write in CVs / free text
# Values = the exact terms that exist in the dataset vocabulary
REAL_WORLD_BRIDGE = {
    # Technology / CS
    "programming":          {"skills": ["Python", "Java", "React"], "interests": ["Technology", "Competitive Programming"]},
    "coding":               {"skills": ["Python", "Java", "React"], "interests": ["Technology", "Competitive Programming"]},
    "software development": {"skills": ["System Design", "Docker", "Kubernetes", "React"], "interests": ["Technology", "Open Source"]},
    "software engineering": {"skills": ["System Design", "Docker", "Java", "React"], "interests": ["Technology"]},
    "web development":      {"skills": ["React", "System Design"], "interests": ["Web Development", "Technology"]},
    "frontend":             {"skills": ["React"], "interests": ["Web Development", "Technology"]},
    "backend":              {"skills": ["Docker", "Kubernetes", "System Design"], "interests": ["Technology", "Cybersecurity"]},
    "devops":               {"skills": ["Ci/Cd", "Docker", "Kubernetes"], "interests": ["Technology", "Open Source"]},
    "artificial intelligence": {"skills": ["Deep Learning", "Pytorch", "Tensorflow", "Nlp"], "interests": ["Artificial Intelligence", "Neural Networks"]},
    "ai":                   {"skills": ["Deep Learning", "Pytorch", "Tensorflow"], "interests": ["Artificial Intelligence", "Neural Networks"]},
    "machine learning":     {"skills": ["Feature Engineering", "Deep Learning", "Mlops", "Tensorflow", "Pytorch"], "interests": ["Artificial Intelligence", "Statistical Modeling", "Neural Networks"]},
    "ml":                   {"skills": ["Feature Engineering", "Mlops", "Pytorch"], "interests": ["Artificial Intelligence", "Statistical Modeling"]},
    "deep learning":        {"skills": ["Deep Learning", "Pytorch", "Tensorflow", "Nlp"], "interests": ["Neural Networks", "Artificial Intelligence"]},
    "neural network":       {"skills": ["Deep Learning", "Pytorch", "Tensorflow"], "interests": ["Neural Networks", "Artificial Intelligence"]},
    "nlp":                  {"skills": ["Nlp", "Deep Learning", "Pytorch"], "interests": ["Artificial Intelligence", "Neural Networks"]},
    "natural language":     {"skills": ["Nlp", "Deep Learning"], "interests": ["Artificial Intelligence"]},
    "data science":         {"skills": ["Data Analysis", "Feature Engineering", "Deep Learning", "Mlops"], "interests": ["Statistical Modeling", "Artificial Intelligence", "Technology"]},
    "data analysis":        {"skills": ["Data Analysis", "Report Writing", "Analytical Thinking"], "interests": ["Statistical Modeling", "Technology"]},
    "data analytics":       {"skills": ["Data Analysis", "Report Writing", "Analytical Thinking"], "interests": ["Statistical Modeling"]},
    "computer vision":      {"skills": ["Deep Learning", "Pytorch", "Tensorflow"], "interests": ["Neural Networks", "Artificial Intelligence"]},
    "cybersecurity":        {"skills": ["System Design", "Python"], "interests": ["Cybersecurity", "Technology"]},
    "cloud":                {"skills": ["Docker", "Kubernetes", "Ci/Cd"], "interests": ["Technology", "Open Source"]},
    "aws":                  {"skills": ["Docker", "Kubernetes", "Ci/Cd"], "interests": ["Technology"]},
    "azure":                {"skills": ["Docker", "Kubernetes", "Ci/Cd"], "interests": ["Technology"]},
    "database":             {"skills": ["Data Analysis", "System Design"], "interests": ["Technology"]},
    "sql":                  {"skills": ["Data Analysis", "Report Writing"], "interests": ["Statistical Modeling", "Technology"]},
    "javascript":           {"skills": ["React", "System Design"], "interests": ["Web Development", "Technology"]},
    "typescript":           {"skills": ["React", "System Design"], "interests": ["Web Development"]},
    "competitive programming": {"skills": ["Python", "Analytical Thinking"], "interests": ["Competitive Programming", "Technology"]},
    "open source":          {"skills": ["Python", "Docker"], "interests": ["Open Source", "Technology"]},
    "github":               {"skills": ["Python", "Docker"], "interests": ["Open Source", "Technology"]},
    "git":                  {"skills": ["Python", "System Design"], "interests": ["Technology"]},
    "agile":                {"skills": ["Project Management", "Teamwork"], "interests": ["Technology", "Business"]},
    "scrum":                {"skills": ["Project Management", "Teamwork"], "interests": ["Technology", "Business"]},
    "xgboost":              {"skills": ["Feature Engineering", "Deep Learning", "Mlops"], "interests": ["Artificial Intelligence", "Statistical Modeling"]},
    "scikit":               {"skills": ["Feature Engineering", "Mlops"], "interests": ["Artificial Intelligence"]},
    "pandas":               {"skills": ["Data Analysis", "Feature Engineering"], "interests": ["Statistical Modeling"]},
    "numpy":                {"skills": ["Data Analysis", "Feature Engineering"], "interests": ["Statistical Modeling"]},
    "streamlit":            {"skills": ["Python", "React"], "interests": ["Technology"]},
    "flask":                {"skills": ["Python", "System Design"], "interests": ["Technology", "Web Development"]},
    "django":               {"skills": ["Python", "System Design"], "interests": ["Technology", "Web Development"]},
    "node":                 {"skills": ["React", "System Design"], "interests": ["Web Development", "Technology"]},
    "api":                  {"skills": ["System Design", "Docker"], "interests": ["Technology"]},

    # Business / Finance
    "finance":              {"skills": ["Report Writing", "Data Analysis", "Ms Office"], "interests": ["Business"]},
    "accounting":           {"skills": ["Report Writing", "Ms Office", "Data Analysis"], "interests": ["Business"]},
    "business analysis":    {"skills": ["Data Analysis", "Report Writing", "Stakeholder Management"], "interests": ["Business"]},
    "financial modeling":   {"skills": ["Data Analysis", "Report Writing", "Ms Office"], "interests": ["Business"]},
    "marketing":            {"skills": ["Presentation Skills", "Communication", "Report Writing"], "interests": ["Business", "Writing"]},
    "sales":                {"skills": ["Communication", "Presentation Skills"], "interests": ["Business"]},
    "management":           {"skills": ["Project Management", "Leadership", "Teamwork"], "interests": ["Business"]},
    "excel":                {"skills": ["Ms Office", "Data Analysis"], "interests": ["Business"]},

    # Healthcare
    "medicine":             {"skills": ["Research", "Analytical Thinking"], "interests": ["Healthcare", "Innovation"]},
    "clinical":             {"skills": ["Research", "Analytical Thinking"], "interests": ["Healthcare"]},
    "pharmacy":             {"skills": ["Research", "Analytical Thinking"], "interests": ["Healthcare"]},
    "medical":              {"skills": ["Research", "Analytical Thinking"], "interests": ["Healthcare"]},
    "nursing":              {"skills": ["Communication", "Teamwork"], "interests": ["Healthcare", "Community Service"]},
    "dentistry":            {"skills": ["Research", "Analytical Thinking"], "interests": ["Healthcare"]},
    "public health":        {"skills": ["Research", "Data Analysis", "Report Writing"], "interests": ["Healthcare", "Community Service"]},
    "therapy":              {"skills": ["Communication", "Research"], "interests": ["Healthcare", "Education"]},

    # Research / Academia
    "research":             {"skills": ["Research", "Analytical Thinking", "Report Writing"], "interests": ["Innovation", "Education"]},
    "statistics":           {"skills": ["Data Analysis", "Analytical Thinking", "Feature Engineering"], "interests": ["Statistical Modeling"]},
    "mathematics":          {"skills": ["Analytical Thinking", "Data Analysis"], "interests": ["Statistical Modeling"]},
    "academia":             {"skills": ["Research", "Report Writing", "Presentation Skills"], "interests": ["Education", "Innovation"]},
    "phd":                  {"skills": ["Research", "Analytical Thinking", "Report Writing"], "interests": ["Innovation", "Education"]},

    # Creative / Media
    "writing":              {"skills": ["Report Writing", "Presentation Skills", "Communication"], "interests": ["Writing", "Arts"]},
    "journalism":           {"skills": ["Report Writing", "Communication", "Presentation Skills"], "interests": ["Writing", "Public Speaking"]},
    "photography":          {"skills": ["Presentation Skills", "Attention To Detail"], "interests": ["Arts"]},
    "design":               {"skills": ["Attention To Detail", "Presentation Skills"], "interests": ["Arts", "Innovation"]},
    "music":                {"skills": ["Presentation Skills", "Teamwork"], "interests": ["Arts"]},
    "film":                 {"skills": ["Presentation Skills", "Communication"], "interests": ["Arts", "Writing"]},
    "acting":               {"skills": ["Communication", "Presentation Skills"], "interests": ["Arts", "Public Speaking"]},

    # Law / Policy
    "law":                  {"skills": ["Research", "Report Writing", "Analytical Thinking"], "interests": ["Business"]},
    "legal":                {"skills": ["Research", "Report Writing", "Analytical Thinking"], "interests": ["Business"]},
    "policy":               {"skills": ["Research", "Report Writing", "Data Analysis"], "interests": ["Business", "Community Service"]},
    "diplomacy":            {"skills": ["Communication", "Presentation Skills", "Research"], "interests": ["Community Service", "Public Speaking"]},

    # Soft skills 
    "leadership":           {"skills": ["Leadership", "Project Management"], "interests": []},
    "communication":        {"skills": ["Communication", "Presentation Skills"], "interests": []},
    "problem solving":      {"skills": ["Problem Solving", "Analytical Thinking", "Critical Thinking"], "interests": []},
    "teamwork":             {"skills": ["Teamwork", "Project Management"], "interests": []},
    "critical thinking":    {"skills": ["Critical Thinking", "Analytical Thinking"], "interests": []},
    "analytical":           {"skills": ["Analytical Thinking", "Data Analysis"], "interests": []},
    "time management":      {"skills": ["Time Management", "Project Management"], "interests": []},
    "attention to detail":  {"skills": ["Attention To Detail"], "interests": []},
    "stakeholder":          {"skills": ["Stakeholder Management", "Project Management", "Communication"], "interests": []},
}

# DOMAIN DETECTION
# These keywords in user text signal a specific domain strongly
DOMAIN_SIGNALS = {
    "Technology": [
        "python", "java", "react", "docker", "kubernetes", "machine learning",
        "deep learning", "tensorflow", "pytorch", "sql", "database", "api",
        "software", "coding", "programming", "developer", "engineer", "ml",
        "ai", "artificial intelligence", "data science", "nlp", "git",
        "github", "aws", "azure", "cloud", "devops", "agile", "algorithm",
        "computer science", "information technology", "xgboost", "scikit",
        "pandas", "numpy", "flask", "django", "node", "javascript",
        "frontend", "backend", "web development", "cybersecurity", "ci/cd",
        "open source", "competitive programming", "streamlit", "neural"
    ],
    "Business": [
        "finance", "accounting", "marketing", "sales", "business", "excel",
        "financial modeling", "investment", "equity", "audit", "tax",
        "commerce", "economics", "mba", "management", "strategy",
        "consulting", "analyst", "crm", "b2b", "revenue", "budget"
    ],
    "Healthcare": [
        "medicine", "medical", "clinical", "pharmacy", "pharmacist", "nursing",
        "nurse", "doctor", "surgery", "dentistry", "dental", "therapy",
        "therapist", "public health", "hospital", "patient", "mbbs",
        "healthcare", "anatomy", "physiology", "pathology", "radiology"
    ],
    "Education & Research": [
        "teaching", "professor", "academia", "research", "phd", "thesis",
        "dissertation", "curriculum", "pedagogy", "counseling", "education",
        "school", "university", "lecture", "publication", "journal"
    ],
    "Creative": [
        "writing", "journalism", "photography", "music", "film", "acting",
        "dance", "art", "design", "creative", "media", "broadcast",
        "publishing", "storytelling", "content", "blog", "podcast"
    ],
    "Law & Policy": [
        "law", "legal", "policy", "legislation", "judiciary", "military",
        "defense", "diplomacy", "international relations", "government",
        "politics", "police", "enforcement", "public relations", "pr"
    ]
}

# DOMAIN - VALID CAREERS 
DOMAIN_CAREERS = {
    "Technology":           ["Software Engineer", "ML Engineer", "Data Entry Operator", "Game Designer", "Research Scientist"],
    "Business":             ["Business Analyst", "Financial Analyst", "Junior Accountant", "Marketing Executive", "Sales Assistant", "Policy Analyst"],
    "Healthcare":           ["Doctor", "Surgeon", "Nurse", "Pharmacist", "Dentist", "Therapist", "Public Health Specialist"],
    "Education & Research": ["Professor", "School Counselor", "Research Scientist"],
    "Creative":             ["Journalist", "Writer", "Publisher", "Broadcaster", "Photographer", "Visual Artist", "Filmmaker", "Musician", "Dancer", "Actor"],
    "Law & Policy":         ["Lawyer", "Judge", "Law Enforcement Officer", "Diplomat", "Politician", "Policy Analyst", "Military Officer", "Public Relations Specialist"],
}


def detect_domain(text: str, user_skills: list = None, user_interests: list = None) -> str | None:
    """
    Returns the most likely domain based on text, skills, and interests.
    Returns None if no domain is clearly detected (low confidence).
    """
    combined = text.lower()
    if user_skills:    combined += " " + " ".join(user_skills).lower()
    if user_interests: combined += " " + " ".join(user_interests).lower()

    scores = {domain: 0 for domain in DOMAIN_SIGNALS}
    for domain, keywords in DOMAIN_SIGNALS.items():
        for kw in keywords:
            import re
            if re.search(r'\b' + re.escape(kw) + r'\b', combined):
                scores[domain] += 1

    best_domain  = max(scores, key=scores.get)
    best_score   = scores[best_domain]
    second_score = sorted(scores.values(), reverse=True)[1]

    # Only return domain if signal is strong enough
    if best_score == 0:
        return None
    if best_score < 2 and (best_score - second_score) < 1:
        return None  # too ambiguous

    return best_domain


def enrich_profile(raw_text: str, skills: list, interests: list) -> tuple:
    """
    Enriches user's extracted skills/interests with dataset-matching vocabulary
    using the REAL_WORLD_BRIDGE.
    Returns (enriched_skills, enriched_interests)
    """
    import re
    text = (raw_text + " " + " ".join(skills) + " " + " ".join(interests)).lower()

    extra_skills    = list(skills)
    extra_interests = list(interests)

    for trigger, additions in REAL_WORLD_BRIDGE.items():
        if re.search(r'\b' + re.escape(trigger) + r'\b', text):
            for sk in additions.get("skills", []):
                if sk not in extra_skills:
                    extra_skills.append(sk)
            for it in additions.get("interests", []):
                if it not in extra_interests:
                    extra_interests.append(it)

    return extra_skills[:30], extra_interests[:25]


def rerank_with_domain(predictions: list, detected_domain: str | None) -> list:
    """
    Given a list of {career, norm, raw, strength} dicts and a detected domain,
    reranks/filters predictions to ensure domain-appropriate careers appear.

    If domain is confidently detected:
      - Careers in the domain are boosted
      - Out-of-domain careers are moved to lower positions
    """
    if not detected_domain or not predictions:
        return predictions

    valid_careers = DOMAIN_CAREERS.get(detected_domain, [])
    if not valid_careers:
        return predictions

    in_domain  = [p for p in predictions if p["career"] in valid_careers]
    out_domain = [p for p in predictions if p["career"] not in valid_careers]

    # If we have at least one in-domain result, prioritize domain results
    if in_domain:
        reranked = in_domain + out_domain
        # Re-normalize the top-3 after reranking
        return reranked[:3]

    return predictions

# SKILL GAP DATABASE + COURSE RECOMMENDATION ENGINE
# CareerML Project — skill_gap_db.py
# For every career in our model, we define the required skills.
# Then for each skill, we link to a real course the user can take.

SKILL_GAP_DB = {
    "Software Engineer": [
        "Python", "Data Structures", "Algorithms", "Git", "APIs",
        "Problem Solving", "Object-Oriented Programming", "SQL", "System Design", "Testing"
    ],
    "ML Engineer": [
        "Python", "Machine Learning", "Deep Learning", "TensorFlow",
        "Statistics", "Data Preprocessing", "Model Evaluation", "SQL", "Feature Engineering", "Git"
    ],
    "Data Entry Operator": [
        "Typing Speed", "Attention To Detail", "Microsoft Excel",
        "Data Accuracy", "Time Management", "Organizational Skills"
    ],
    "Business Analyst": [
        "Data Analysis", "SQL", "Excel", "Communication", "Problem Solving",
        "Requirements Gathering", "Presentation Skills", "Project Management"
    ],
    "Financial Analyst": [
        "Financial Modeling", "Excel", "Accounting", "Statistics",
        "Data Analysis", "Communication", "Research", "Report Writing"
    ],
    "Junior Accountant": [
        "Bookkeeping", "Excel", "Accounting", "Tax Preparation",
        "Attention To Detail", "Financial Reporting", "Auditing"
    ],
    "Marketing Executive": [
        "Digital Marketing", "Social Media", "SEO", "Content Creation",
        "Analytics", "Communication", "Brand Management", "Market Research"
    ],
    "Sales Assistant": [
        "Communication", "Customer Service", "Negotiation",
        "CRM Software", "Product Knowledge", "Time Management"
    ],
    "Doctor": [
        "Clinical Knowledge", "Diagnosis", "Patient Care", "Medical Ethics",
        "Communication", "Research", "Emergency Medicine", "Pharmacology"
    ],
    "Surgeon": [
        "Surgical Skills", "Anatomy", "Clinical Knowledge", "Hand-Eye Coordination",
        "Crisis Management", "Medical Ethics", "Research", "Patient Care"
    ],
    "Nurse": [
        "Patient Care", "Clinical Skills", "Communication", "Empathy",
        "Medical Knowledge", "Teamwork", "Critical Thinking", "Documentation"
    ],
    "Pharmacist": [
        "Pharmacology", "Drug Interactions", "Patient Counseling",
        "Attention To Detail", "Medical Knowledge", "Communication", "Inventory Management"
    ],
    "Dentist": [
        "Dental Procedures", "Patient Care", "Anatomy", "Attention To Detail",
        "Communication", "Medical Ethics", "Radiology", "Anesthesiology"
    ],
    "Therapist": [
        "Counseling", "Empathy", "Active Listening", "Psychology",
        "CBT Techniques", "Communication", "Documentation", "Ethics"
    ],
    "School Counselor": [
        "Counseling", "Empathy", "Communication", "Child Development",
        "Career Guidance", "Conflict Resolution", "Active Listening", "Psychology"
    ],
    "Professor": [
        "Subject Expertise", "Research", "Teaching", "Academic Writing",
        "Communication", "Curriculum Design", "Critical Thinking", "Mentoring"
    ],
    "Research Scientist": [
        "Research Methodology", "Statistical Analysis", "Python", "Data Collection",
        "Academic Writing", "Critical Thinking", "Lab Skills", "Peer Review"
    ],
    "Public Health Specialist": [
        "Epidemiology", "Data Analysis", "Public Policy", "Communication",
        "Research", "Community Outreach", "Health Education", "Statistics"
    ],
    "Journalist": [
        "Writing", "Research", "Interviewing", "Editing",
        "Communication", "Social Media", "Investigative Skills", "Ethics"
    ],
    "Writer": [
        "Creative Writing", "Research", "Editing", "Grammar",
        "Storytelling", "Publishing Knowledge", "Social Media", "Self-Discipline"
    ],
    "Publisher": [
        "Editing", "Project Management", "Market Research", "Communication",
        "Contract Negotiation", "Literary Knowledge", "Digital Publishing", "Networking"
    ],
    "Broadcaster": [
        "Public Speaking", "Research", "Communication", "Audio/Video Production",
        "Social Media", "Interviewing", "News Writing", "Adaptability"
    ],
    "Photographer": [
        "Photography", "Composition", "Photo Editing Software", "Lighting",
        "Creativity", "Client Management", "Color Theory", "Business Skills"
    ],
    "Visual Artist": [
        "Drawing", "Painting", "Color Theory", "Creativity",
        "Digital Art", "Composition", "Art History", "Portfolio Development"
    ],
    "Game Designer": [
        "Game Design", "Storytelling", "Unity Or Unreal Engine", "Programming",
        "User Experience", "Creativity", "Level Design", "Teamwork"
    ],
    "Filmmaker": [
        "Cinematography", "Storytelling", "Video Editing", "Directing",
        "Screenwriting", "Production Management", "Sound Design", "Creativity"
    ],
    "Musician": [
        "Music Theory", "Instrument Proficiency", "Composition",
        "Performance", "Music Production", "Ear Training", "Collaboration"
    ],
    "Dancer": [
        "Dance Techniques", "Rhythm", "Flexibility", "Performance",
        "Choreography", "Physical Fitness", "Music Knowledge", "Discipline"
    ],
    "Actor": [
        "Acting Techniques", "Voice Modulation", "Improvisation",
        "Script Analysis", "Physical Expression", "Memorization", "Stage Presence"
    ],
    "Lawyer": [
        "Legal Research", "Critical Thinking", "Communication", "Argumentation",
        "Legal Writing", "Ethics", "Negotiation", "Case Analysis"
    ],
    "Judge": [
        "Legal Knowledge", "Critical Thinking", "Impartiality",
        "Communication", "Ethics", "Research", "Decision Making"
    ],
    "Law Enforcement Officer": [
        "Physical Fitness", "Communication", "Crisis Management", "Ethics",
        "Investigation", "First Aid", "Teamwork", "Conflict Resolution"
    ],
    "Diplomat": [
        "International Relations", "Communication", "Negotiation", "Foreign Languages",
        "Cultural Awareness", "Research", "Public Speaking", "Political Science"
    ],
    "Politician": [
        "Public Speaking", "Leadership", "Communication", "Policy Knowledge",
        "Networking", "Negotiation", "Ethics", "Crisis Management"
    ],
    "Policy Analyst": [
        "Research", "Data Analysis", "Policy Writing", "Communication",
        "Critical Thinking", "Statistics", "Economics", "Report Writing"
    ],
    "Military Officer": [
        "Leadership", "Physical Fitness", "Strategic Thinking", "Communication",
        "Crisis Management", "Teamwork", "Ethics", "Decision Making"
    ],
    "Public Relations Specialist": [
        "Communication", "Writing", "Social Media", "Media Relations",
        "Crisis Management", "Networking", "Brand Management", "Research"
    ],
}

COURSE_LOOKUP = {
    "Python":                  ("Python for Everybody – Coursera",            "https://www.coursera.org/specializations/python"),
    "Machine Learning":        ("Machine Learning Specialization – Coursera",  "https://www.coursera.org/specializations/machine-learning-introduction"),
    "Deep Learning":           ("Deep Learning Specialization – Coursera",     "https://www.coursera.org/specializations/deep-learning"),
    "TensorFlow":              ("TensorFlow Developer Certificate – Google",   "https://www.coursera.org/professional-certificates/tensorflow-in-practice"),
    "SQL":                     ("SQL for Data Science – Coursera",             "https://www.coursera.org/learn/sql-for-data-science"),
    "Data Analysis":           ("Google Data Analytics Certificate",           "https://www.coursera.org/professional-certificates/google-data-analytics"),
    "Statistics":              ("Statistics with Python – Coursera",           "https://www.coursera.org/specializations/statistics-with-python"),
    "Statistical Analysis":    ("Statistics with Python – Coursera",           "https://www.coursera.org/specializations/statistics-with-python"),
    "Excel":                   ("Excel Skills for Business – Coursera",        "https://www.coursera.org/specializations/excel"),
    "Microsoft Excel":         ("Excel Skills for Business – Coursera",        "https://www.coursera.org/specializations/excel"),
    "Communication":           ("Improving Communication Skills – Coursera",   "https://www.coursera.org/learn/wharton-communication-skills"),
    "Research":                ("Research Methods – FutureLearn",              "https://www.futurelearn.com/courses/research-methods"),
    "Research Methodology":    ("Research Methods – FutureLearn",              "https://www.futurelearn.com/courses/research-methods"),
    "Critical Thinking":       ("Critical Thinking & Problem Solving – edX",   "https://www.edx.org/learn/critical-thinking"),
    "Digital Marketing":       ("Google Digital Marketing & E-commerce",       "https://www.coursera.org/professional-certificates/google-digital-marketing-ecommerce"),
    "Social Media":            ("Social Media Marketing – Meta",               "https://www.coursera.org/professional-certificates/facebook-social-media-marketing"),
    "SEO":                     ("SEO Fundamentals – Semrush Academy",          "https://www.semrush.com/academy/"),
    "Content Creation":        ("Content Strategy – Northwestern/Coursera",    "https://www.coursera.org/specializations/content-strategy"),
    "Financial Modeling":      ("Financial Modeling – Udemy",                  "https://www.udemy.com/course/the-complete-financial-analyst-course/"),
    "Accounting":              ("Accounting Fundamentals – edX",               "https://www.edx.org/learn/accounting"),
    "Bookkeeping":             ("Bookkeeping Basics – Coursera",               "https://www.coursera.org/learn/bookkeeping-basics"),
    "Financial Reporting":     ("Financial Reporting – Coursera",              "https://www.coursera.org/learn/financial-reporting"),
    "Project Management":      ("Google Project Management Certificate",       "https://www.coursera.org/professional-certificates/google-project-management"),
    "Creative Writing":        ("Creative Writing Specialization – Coursera",  "https://www.coursera.org/specializations/creative-writing"),
    "Writing":                 ("English Composition – Coursera",              "https://www.coursera.org/learn/writing"),
    "Editing":                 ("Writing and Editing – Coursera",              "https://www.coursera.org/specializations/writing-editing"),
    "Counseling":              ("Foundations of Counseling – Alison",          "https://alison.com/courses/counselling-skills"),
    "Psychology":              ("Introduction to Psychology – Coursera",       "https://www.coursera.org/learn/introduction-psychology"),
    "CBT Techniques":          ("CBT Specialization – Coursera",               "https://www.coursera.org/specializations/cognitive-behavioral-therapy"),
    "Empathy":                 ("Emotional Intelligence – edX",                "https://www.edx.org/learn/emotional-intelligence"),
    "Public Speaking":         ("Dynamic Public Speaking – Coursera",          "https://www.coursera.org/specializations/public-speaking"),
    "Legal Research":          ("Introduction to English Common Law – Coursera","https://www.coursera.org/learn/intro-common-law"),
    "Legal Writing":           ("Legal Writing – Coursera",                    "https://www.coursera.org/learn/legal-writing"),
    "Negotiation":             ("Successful Negotiation – Coursera",           "https://www.coursera.org/learn/negotiation"),
    "Leadership":              ("Leadership and Management – Coursera",        "https://www.coursera.org/specializations/leadership-development-for-engineers"),
    "Ethics":                  ("Ethics in Technology – edX",                  "https://www.edx.org/learn/ethics"),
    "Photography":             ("Photography Basics – Coursera",               "https://www.coursera.org/learn/photography-basics"),
    "Photo Editing Software":  ("Adobe Photoshop – Udemy",                     "https://www.udemy.com/course/adobe-photoshop-cc-essentials-training-course/"),
    "Color Theory":            ("Graphic Design Specialization – Coursera",    "https://www.coursera.org/specializations/graphic-design"),
    "Music Theory":            ("Musicianship – Berklee/Coursera",             "https://www.coursera.org/learn/musicianship-chords"),
    "Music Production":        ("Music Production – Coursera",                 "https://www.coursera.org/learn/music-production"),
    "Acting Techniques":       ("Introduction to Acting – Udemy",              "https://www.udemy.com/course/the-complete-acting-course/"),
    "Screenwriting":           ("Screenwriting – Coursera",                    "https://www.coursera.org/specializations/screenwriting"),
    "Video Editing":           ("Video Editing with DaVinci Resolve – Udemy",  "https://www.udemy.com/course/davinci-resolve-video-editing/"),
    "Game Design":             ("Game Design and Development – Coursera",      "https://www.coursera.org/specializations/game-development"),
    "Unity Or Unreal Engine":  ("Unreal Engine – Udemy",                       "https://www.udemy.com/course/unrealcourse/"),
    "Epidemiology":            ("Epidemiology: The Basic Science – Coursera",  "https://www.coursera.org/learn/epidemiology"),
    "Public Policy":           ("Public Policy – edX",                         "https://www.edx.org/learn/public-policy"),
    "International Relations": ("International Relations – Coursera",          "https://www.coursera.org/learn/international-relations"),
    "Academic Writing":        ("Academic Writing – Coursera",                 "https://www.coursera.org/learn/academic-writing"),
    "Data Structures":         ("Algorithms Specialization – Coursera",        "https://www.coursera.org/specializations/algorithms"),
    "Algorithms":              ("Algorithms Specialization – Coursera",        "https://www.coursera.org/specializations/algorithms"),
    "Git":                     ("Version Control with Git – Coursera",         "https://www.coursera.org/learn/version-control-with-git"),
    "System Design":           ("System Design – Educative",                   "https://www.educative.io/courses/grokking-the-system-design-interview"),
    "Feature Engineering":     ("Feature Engineering – Kaggle",                "https://www.kaggle.com/learn/feature-engineering"),
    "Data Preprocessing":      ("Data Preprocessing – Kaggle",                 "https://www.kaggle.com/learn/data-cleaning"),
    "Model Evaluation":        ("ML Evaluation Metrics – Coursera",            "https://www.coursera.org/learn/machine-learning"),
    "Patient Care":            ("Patient Care Fundamentals – Coursera",        "https://www.coursera.org/learn/patient-safety"),
    "Pharmacology":            ("Drug Development – Coursera",                 "https://www.coursera.org/learn/drug-development"),
    "Anatomy":                 ("Anatomy Specialization – Coursera",           "https://www.coursera.org/specializations/anatomy"),
    "First Aid":               ("First Aid – Red Cross",                       "https://www.redcross.org/take-a-class/first-aid"),
    "Physical Fitness":        ("Exercise Physiology – Coursera",              "https://www.coursera.org/learn/exercise"),
    "Foreign Languages":       ("Language Learning – Duolingo",                "https://www.duolingo.com"),
    "Cinematography":          ("Filmmaking – Coursera",                       "https://www.coursera.org/specializations/filmmaking"),
    "Drawing":                 ("Drawing – Coursera",                          "https://www.coursera.org/learn/design-thinking-innovation"),
    "Storytelling":            ("Storytelling & Influence – Coursera",         "https://www.coursera.org/learn/storytelling-and-influencing"),
    "Brand Management":        ("Brand Management – Coursera",                 "https://www.coursera.org/learn/brand-management"),
    "Market Research":         ("Market Research – Coursera",                  "https://www.coursera.org/learn/market-research"),
    "Report Writing":          ("Report Writing – FutureLearn",                "https://www.futurelearn.com/courses/professional-writing"),
    "Economics":               ("Microeconomics – Coursera",                   "https://www.coursera.org/learn/microeconomics"),
    "Networking":              ("Networking – Coursera",                       "https://www.coursera.org/learn/computer-networking"),
    "Teaching":                ("Teaching & Learning – Coursera",              "https://www.coursera.org/learn/teach-online"),
    "Curriculum Design":       ("Curriculum Design – Coursera",                "https://www.coursera.org/learn/curriculum"),
    "Mentoring":               ("Mentoring – Coursera",                        "https://www.coursera.org/learn/mentoring"),
    "Child Development":       ("Child Development – Coursera",                "https://www.coursera.org/learn/child-development"),
    "Career Guidance":         ("Career Counseling – Alison",                  "https://alison.com/tag/career-counselling"),
    "Conflict Resolution":     ("Conflict Management – Coursera",              "https://www.coursera.org/specializations/conflict-management"),
    "Crisis Management":       ("Crisis Management – edX",                     "https://www.edx.org/learn/crisis-management"),
    "Documentation":           ("Technical Writing – Google/Coursera",         "https://www.coursera.org/learn/technical-writing"),
    "Decision Making":         ("Decision Making – Coursera",                  "https://www.coursera.org/learn/decision-making"),
    "Strategic Thinking":      ("Strategic Planning – Coursera",               "https://www.coursera.org/learn/strategic-planning"),
    "Investigative Skills":    ("Investigative Journalism – Coursera",         "https://www.coursera.org/learn/journalism"),
    "Interviewing":            ("Interviewing Techniques – LinkedIn Learning",  "https://www.linkedin.com/learning/topics/interviewing"),
    "News Writing":            ("Journalism for Social Change – edX",          "https://www.edx.org/learn/journalism"),
    "Media Relations":         ("Public Relations – Coursera",                 "https://www.coursera.org/learn/public-relations"),
    "Community Outreach":      ("Community Engagement – Coursera",             "https://www.coursera.org/learn/community-engagement"),
    "Health Education":        ("Health Communication – Coursera",             "https://www.coursera.org/learn/health-communication"),
    "Lab Skills":              ("Lab Safety – Coursera",                       "https://www.coursera.org/learn/lab-safety"),
    "Peer Review":             ("Academic Writing – Coursera",                 "https://www.coursera.org/learn/academic-writing"),
    "Attention To Detail":     ("Quality Management – Coursera",               "https://www.coursera.org/learn/quality-management"),
    "Time Management":         ("Work Smarter Not Harder – Coursera",          "https://www.coursera.org/learn/work-smarter-not-harder"),
    "Teamwork":                ("Teamwork & Collaboration – Coursera",         "https://www.coursera.org/learn/teamwork-skills"),
    "Problem Solving":         ("Critical Thinking & Problem Solving – edX",   "https://www.edx.org/learn/critical-thinking"),
    "Creativity":              ("Creative Thinking – Coursera",                "https://www.coursera.org/learn/creative-thinking-techniques"),
    "Customer Service":        ("Customer Service – Coursera",                 "https://www.coursera.org/learn/customer-service-fundamentals"),
    "Adaptability":            ("Building Adaptability – Coursera",            "https://www.coursic.org"),
}


def get_skill_gap(user_skills: list, career: str) -> list:
    required = set(SKILL_GAP_DB.get(career, []))
    user_set  = {s.strip().title() for s in user_skills}
    return sorted(required - user_set)


def get_courses(missing_skills: list) -> list:
    result = []
    for skill in missing_skills:
        if skill in COURSE_LOOKUP:
            name, url = COURSE_LOOKUP[skill]
            result.append((skill, name, url))
        else:
            result.append((skill,
                           f"Search '{skill}' on Coursera",
                           f"https://www.coursera.org/search?query={skill.replace(' ', '+')}"))
    return result

import json
import random
import csv

# Load data from files
def load_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_company_names(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row[1].strip() for row in reader if row]

def load_locations(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [
            f"{row['name_en'].strip()}, {row['country'].strip()}"
            for row in reader
            if row.get("name_en") and row.get("country")
        ]

# Load datasets
first_names = load_file_lines('./datasets/first_names.txt')
last_names = load_file_lines('./datasets/last_names.txt')
emails = load_file_lines('./datasets/emails.txt')
job_titles = load_file_lines('./datasets/job-titles.txt')
colleges = load_file_lines('./datasets/all_US_colleges.txt')
majors = load_json('./datasets/all_majors.json')
skills = load_file_lines('./datasets/linkedin_skills.txt')
companies = load_company_names('./datasets/companies.csv')
locations = load_locations('./datasets/world_cities_geoname.csv')

# Generate a random profile
def generate_profile():
    first = random.choice(first_names)
    last = random.choice(last_names).capitalize()

    # Phone number
    area_code = random.randint(200, 999)
    middle = random.randint(100, 999)
    last_four = random.randint(1000, 9999)
    phone = f"({area_code}) {middle}-{last_four}"

    # Education
    degree_types = ["Associates Degree", "Bachelors Degree", "Masters Degree", 
                    "Doctorate", "Certificate", "Vocational Degree"]
    max_degrees = random.randint(1, 5)
    used_once_only = {"Associates Degree", "Doctorate"}
    selected_degrees = []
    education = []

    while len(education) < max_degrees:
        degree = random.choice(degree_types)
        if degree in used_once_only and degree in selected_degrees:
            continue
        selected_degrees.append(degree)
        education.append({
            "degree_type": degree,
            "major": random.choice(majors),
            "college": random.choice(colleges)
        })

    # Job history
    num_jobs = random.randint(1, 5)
    job_history = []
    for _ in range(num_jobs):
        job_title = random.choice(job_titles).title()
        company = random.choice(companies)
        years = random.randint(1, 15)
        job_history.append({
            "job_title": job_title,
            "company": company,
            "years_in_role": years
        })

    # Location
    location = random.choice(locations)

    profile = {
        "first_name": first,
        "last_name": last,
        "email": random.choice(emails),
        "phone_number": phone,
        "location": location,
        "education": education,
        "job_history": job_history,
        "skills": random.sample(skills, k=random.randint(8, 30))
    }
    return profile

# Main function to create profiles
def create_profiles(num_profiles=100, output_path='./applicant_profiles.json'):
    profiles = [generate_profile() for _ in range(num_profiles)]
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2)
    print(f"{num_profiles} profiles saved to {output_path}")

# Example usage:
create_profiles(num_profiles=50)

import csv
import json

input_file_path = 'datasets/fake_job_postings.csv'  # Update path if needed
output_file_path = 'filtered_jobs.json'

required_fields = {
    "company_profile": "about_us",
    "description": "description",
    "requirements": "requirements",
    "telecommuting": "telecommuting"
}

filtered_jobs = []

with open(input_file_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("telecommuting") == "0":
            if all(row.get(field) for field in required_fields if field != "telecommuting"):
                job_entry = {
                    "about_us": row["company_profile"],
                    "description": row["description"],
                    "requirements": row["requirements"]
                }
                filtered_jobs.append(job_entry)

with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(filtered_jobs, json_file, indent=4)

print(f"Filtered jobs saved to {output_file_path}")

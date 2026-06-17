SELECT source_id, job_title, company, description
FROM jobs
WHERE tech_stack IS NULL OR TRIM(tech_stack) = ''
LIMIT :batch_size
SELECT source_id, tech_stack FROM jobs
WHERE tech_stack IS NOT NULL
AND TRIM(tech_stack) != ''
AND source_id > last_sid
ORDER BY source_id
LIMIT :batch_size
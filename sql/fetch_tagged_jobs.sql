SELECT source_id, tech_stack FROM jobs
WHERE tech_stack IS NOT NULL
AND TRIM(tech_stack) != ''
AND CAST(source_id AS INTEGER) > :last_sid
ORDER BY source_id
LIMIT :batch_size
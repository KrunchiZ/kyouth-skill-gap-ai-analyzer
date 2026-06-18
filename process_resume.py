



# ---------------------------------------------------------------------------
# CLI smoke-test  (uv run resume_extractor.py resume.txt)
# ---------------------------------------------------------------------------
 
def main():
	skills = extract_resume_skills(RESUME_PATH)

	print(f"\nExtracted {len(skills)} skills:")
	for s in sorted(skills):
		print(f"  {s}")



import sys; sys.path.insert(0, '.')
from skills.skill_loader import list_builtin_skills
skills = list_builtin_skills()
print(f'Total skills: {len(skills)}')
for s in skills:
    print(f"  [{s['agent']:12}] {s['name']:30} triggers: {len(s['triggers'])}")

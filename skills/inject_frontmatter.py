"""Script to inject proper frontmatter into all extracted skill SKILL.md files."""
import re, yaml
from pathlib import Path

skills_base = Path('d:/Gemini Playgroun/vscodeground/chatbot/backend/skills/builtin')

rewrites = {
    'create-pdf': {
        'name': 'create-pdf', 'agent': 'document',
        'triggers': ['create pdf','generate pdf','make pdf','generate report','write report','export as pdf','pdf report','merge pdf','split pdf']
    },
    'create-docx': {
        'name': 'create-docx', 'agent': 'document',
        'triggers': ['create word','word doc','word document','create docx','generate docx','export to word','write memo','write letter']
    },
    'create-pptx': {
        'name': 'create-pptx', 'agent': 'document',
        'triggers': ['create presentation','make slides','make powerpoint','create pptx','generate slides','slide deck','pitch deck']
    },
    'spreadsheet-analyst': {
        'name': 'spreadsheet-analyst', 'agent': 'data',
        'triggers': ['create excel','create spreadsheet','xlsx file','excel file','analyze excel','pivot data','cross tabulate','deep excel analysis']
    },
    'read-codebase': {
        'name': 'read-codebase', 'agent': 'shell',
        'triggers': ['explore repo','read codebase','summarize project','read file','explore codebase','what does this project do','explain the code']
    },
    'frontend-design': {
        'name': 'frontend-design', 'agent': 'code',
        'triggers': ['create webpage','build website','design landing page','create ui','build frontend','html page','web component','react component']
    },
    'product-knowledge': {
        'name': 'product-knowledge', 'agent': 'chat',
        'triggers': ['about agentx','what can you do','your capabilities','agentx features']
    },
    'web-artifacts-builder': {
        'name': 'web-artifacts-builder', 'agent': 'code',
        'triggers': ['create artifact','build artifact','interactive artifact','web artifact']
    },
    'algorithmic-art': {
        'name': 'algorithmic-art', 'agent': 'code',
        'triggers': ['create art','algorithmic art','generative art','create animation','canvas art']
    },
    'learn': {
        'name': 'learn', 'agent': 'chat',
        'triggers': ['teach me','explain this topic','learning plan','study guide','how does this work']
    },
}

def inject_metadata(skill_dir_name, meta):
    skill_file = skills_base / skill_dir_name / 'SKILL.md'
    if not skill_file.exists():
        print(f'SKIP (no SKILL.md): {skill_dir_name}')
        return
    content = skill_file.read_text(encoding='utf-8', errors='replace')
    body = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL).strip()
    m = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
    orig_desc = ''
    if m:
        try:
            orig_meta = yaml.safe_load(m.group(1))
            orig_desc = str(orig_meta.get('description', '')).strip()
        except:
            pass
    desc_lines = orig_desc[:500].replace('\n', ' ')
    triggers_yaml = '\n'.join(f'    - "{t}"' for t in meta['triggers'])
    new_content = f'---\nname: {meta["name"]}\ndescription: >\n  {desc_lines}\nmetadata:\n  agent: {meta["agent"]}\n  triggers:\n{triggers_yaml}\n---\n\n{body}\n'
    skill_file.write_text(new_content, encoding='utf-8')
    print(f'Updated: {skill_dir_name}')

for skill_dir, meta in rewrites.items():
    inject_metadata(skill_dir, meta)
print('Done!')

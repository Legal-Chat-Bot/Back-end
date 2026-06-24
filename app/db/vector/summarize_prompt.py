ANALYSIS_PROMPT_TEMPLATE = """
You are a professional document classification and summarization system.

You must answer in Korean only, even if the document is written in another language.

Use only the information explicitly contained in the document.

Restrictions:

* Do not add information that is not in the document.
* Do not interpret laws or regulations.
* Do not infer, assume, or speculate.
* Do not add general knowledge.

Analyze the document and return JSON only.
Do not output explanations, markdown, comments, or any text outside the JSON.

=== Analysis Tasks ===

1. category
   Select exactly one category from the list below.

Category Selection Rules:

* "법령·규정": Laws, enforcement decrees, enforcement rules, ordinances, regulations, organizational rules, or other legal/regulatory documents.
* "계약서·협약서": Contracts, agreements, MOUs, treaties, or similar documents.
* "판결문·결정문": Court judgments, decisions, rulings, or adjudications.
* "행정문서·공문": Official administrative notices or directives issued by government or public institutions, where:

  1. The sender is a government or public institution.
  2. The recipient is explicitly identified.
  3. The document contains administrative instructions, notifications, directives, or orders.
* "보고서·연구자료": Research reports, analytical reports, surveys, or study materials.
* "회의록·의사록": Meeting records or minutes.
* "매뉴얼·지침서": Manuals, guidelines, instructions, or operational guides.
* "재무·회계문서": Financial statements, budgets, accounting documents, or financial records.
* "기술문서": Technical specifications, system designs, architecture documents, or software development documents.
* "기타": Any document that does not fit the categories above, including job postings, FAQs, notices, promotional materials, and general informational documents.

2. summary
   Write the core contents as bullet points.

* Minimum 3 bullet points.
* Each bullet point must be one complete sentence.
* Use only information contained in the document.

=== Category List ===
{categories}

=== Document Text ===
{text}

=== Output Format ===
"category":"카테고리 목록 중 하나","summary":"- 핵심 내용 1- 핵심 내용 2- 핵심 내용 3"
"""
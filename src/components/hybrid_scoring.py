# hybrid_startup_analyzer.py

import google.generativeai as genai,dotenv,os
from typing import Dict, Any, List


try:
    from google.colab import userdata
    API_KEY = userdata.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=API_KEY)
except ImportError:
    # If not in Colab, use an environment variable or hardcode it
    # For security, it's better to use environment variables
    import os
    # os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


# [PASTE ALL 8 MATHEMATICAL SCORING FUNCTIONS HERE]
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# --- Category 1: Founding Team ---
def calculate_team_score(team_data: Dict[str, Any]) -> float:
    """Calculates the score for the founding team."""
    scores = {}
    experience_score = (team_data.get('relevant_experience_years', 0) / 10)
    for outcome in team_data.get('prior_startup_outcomes', []):
        if outcome == 'exit': experience_score += 2.5
        elif outcome == 'profitable': experience_score += 1.5
    scores['experience'] = min(experience_score, 10)
    commitment_score = (team_data.get('founder_investment_percentage', 0) / 5.0) * 5
    if team_data.get('esop_pool_percentage', 0) > 15: commitment_score += 2
    scores['commitment'] = min(commitment_score, 10)
    scores['completeness'] = min(len(team_data.get('key_roles_covered', [])) * 2.5, 10)
    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 2: Market ---
def calculate_market_score(market_data: Dict[str, Any]) -> float:
    """Calculates the score for the market."""
    scores = {}
    som = market_data.get('som_usd_millions', 0)
    if som < 5: scores['size'] = 2
    elif 5 <= som < 20: scores['size'] = 5
    elif 20 <= som < 100: scores['size'] = 8
    else: scores['size'] = 10
    cagr = market_data.get('market_cagr_percentage', 0)
    if cagr < 5: scores['growth'] = 2
    elif 5 <= cagr < 15: scores['growth'] = 6
    else: scores['growth'] = 10
    scores['competition'] = max(10 - market_data.get('funded_competitors', 0) * 1.5, 0)
    moat_map = {'none': 2, 'brand': 5, 'ip': 7, 'network_effects': 9}
    scores['moat'] = moat_map.get(market_data.get('moat_strength', 'none'), 2)
    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 3: Product & Technology ---
def calculate_product_score(product_data: Dict[str, Any]) -> float:
    """Calculates the score for the product."""
    scores = {}
    scores['problem_fit'] = 9 if product_data.get('problem_solution_fit') == 'painkiller' else 5
    stage_map = {'idea': 2, 'mvp': 5, 'scaling': 8}
    scores['stage'] = stage_map.get(product_data.get('product_stage', 'idea'), 2)
    tech_map = {'standard_tech': 4, 'unique_data': 8, 'proprietary_ai': 9}
    scores['tech_defensibility'] = tech_map.get(product_data.get('tech_defensibility', 'standard_tech'), 4)
    scores['adoption'] = 10 - product_data.get('adoption_barrier_score', 5)
    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 4: Business Model & Unit Economics ---
def calculate_economics_score(economics_data: Dict[str, Any]) -> float:
    """Calculates the score for the business model."""
    scores = {}
    ltv_cac = economics_data.get('ltv_to_cac_ratio', 0)
    if ltv_cac < 1: scores['ltv_cac'] = 0
    elif 1 <= ltv_cac < 2: scores['ltv_cac'] = 4
    elif 2 <= ltv_cac < 3: scores['ltv_cac'] = 7
    else: scores['ltv_cac'] = 10
    margin = economics_data.get('gross_margin_percentage', 0)
    if margin < 20: scores['margin'] = 2
    elif 20 <= margin < 50: scores['margin'] = 5
    elif 50 <= margin < 80: scores['margin'] = 8
    else: scores['margin'] = 10
    payback = economics_data.get('cac_payback_months', 24)
    if payback > 18: scores['payback'] = 2
    elif 12 < payback <= 18: scores['payback'] = 5
    elif 6 < payback <= 12: scores['payback'] = 8
    else: scores['payback'] = 10
    runway = economics_data.get('runway_months', 0)
    if runway < 6: scores['runway'] = 1
    elif 6 <= runway < 12: scores['runway'] = 4
    elif 12 <= runway < 18: scores['runway'] = 7
    else: scores['runway'] = 10
    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 5: Traction & Metrics ---
def calculate_traction_score(traction_data: Dict[str, Any]) -> float:
    """Calculates the score for startup traction."""
    scores = {}
    growth = traction_data.get('mom_growth_rate_percentage', 0)
    if growth < 5: scores['growth'] = 2
    elif 5 <= growth < 10: scores['growth'] = 5
    elif 10 <= growth < 20: scores['growth'] = 8
    else: scores['growth'] = 10
    retention = traction_data.get('net_revenue_retention', 0)
    if retention < 80: scores['retention'] = 2
    elif 80 <= retention < 100: scores['retention'] = 6
    else: scores['retention'] = 10
    scores['validation'] = 9 if traction_data.get('has_marquee_partner') else 4
    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 6: Financials ---
def calculate_financials_score(financials_data: Dict[str, Any]) -> float:
    """Calculates the score for the company's financials."""
    scores = {}
    scores['realism'] = financials_data.get('projection_realism_score', 5)
    timeline = financials_data.get('profitability_timeline_years', 5)
    if timeline > 5: scores['timeline'] = 2
    elif 3 < timeline <= 5: scores['timeline'] = 5
    elif 1 < timeline <= 3: scores['timeline'] = 8
    else: scores['timeline'] = 10
    debt_ratio = financials_data.get('debt_to_equity_ratio', 0.5)
    scores['health'] = max(10 - (debt_ratio * 10), 0)
    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 7: Risks ---
def calculate_risk_score(risk_data: List[Dict[str, int]]) -> float:
    """Calculates a risk score, where a higher score means lower risk."""
    if not risk_data: return 10.0
    risk_scores = [r['likelihood'] * r['impact'] for r in risk_data]
    average_risk_score = sum(risk_scores) / len(risk_scores)
    inverted_score = 10 - ((average_risk_score - 1) / 24) * 10
    return max(inverted_score, 0)

# --- Category 8: Growth Potential ---
def calculate_growth_potential_score(growth_data: Dict[str, Any]) -> float:
    """Calculates the score for growth potential."""
    scores = {}
    scalability_map = {'low': 3, 'medium': 6, 'high': 9}
    scores['scalability'] = scalability_map.get(growth_data.get('scalability_type', 'low'), 3)
    paths = growth_data.get('expansion_paths', 0)
    if paths == 1: scores['expansion'] = 4
    elif paths == 2: scores['expansion'] = 7
    elif paths >= 3: scores['expansion'] = 9
    else: scores['expansion'] = 1
    acquirers = growth_data.get('potential_acquirers', 0)
    if acquirers <= 2: scores['exit_path'] = 3
    elif 3 <= acquirers <= 5: scores['exit_path'] = 6
    else: scores['exit_path'] = 9
    return sum(scores.values()) / len(scores) if scores else 0

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# [END OF MATHEMATICAL SCORING FUNCTIONS]


# --- Generative AI Analysis Function ---
def generate_qualitative_analysis(unstructured_text: str, scores: Dict[str, float], final_score: float) -> str:
    """
    Uses a generative AI model to create a qualitative investment memo.

    Args:
        unstructured_text: The raw text from the pitch deck, transcripts, etc.
        scores: The dictionary of pre-calculated quantitative scores.
        final_score: The final weighted score.

    Returns:
        A string containing the AI-generated investment memorandum.
    """
    model = genai.GenerativeModel('gemini-pro')

    prompt = f"""
    As an experienced venture capital analyst, your task is to write a concise and insightful investment memorandum.
    You will be provided with raw text from a startup's documents and a set of pre-calculated quantitative scores.
    Your analysis must be grounded in these facts.

    **Quantitative Scores (out of 10):**
    - Final Weighted Score: {final_score:.2f}
    - Team: {scores.get('team', 0):.2f}
    - Market: {scores.get('market', 0):.2f}
    - Product: {scores.get('product', 0):.2f}
    - Unit Economics: {scores.get('economics', 0):.2f}
    - Traction: {scores.get('traction', 0):.2f}
    - Financials: {scores.get('financials', 0):.2f}
    - Risk (10 is lowest risk): {scores.get('risk', 0):.2f}
    - Growth Potential: {scores.get('growth_potential', 0):.2f}

    **Raw Document Text:**
    ---
    {unstructured_text}
    ---

    **Your Task:**
    Write the investment memorandum following this structure:

    **1. Executive Summary:**
       - Start with a one-sentence summary of the company.
       - Briefly state the overall recommendation (e.g., "Recommend for second look," "Pass," "Strongly recommend").
       - Mention the final weighted score and highlight the 1-2 strongest and weakest areas based on the quantitative scores.

    **2. Strengths:**
       - Write 2-3 paragraphs detailing the most promising aspects of the startup.
       - For each point, refer to the relevant category score and use evidence or quotes from the raw text to support your claim.
       - Example: "The founding team is a significant asset, scoring an impressive {scores.get('team', 0):.2f}/10. The raw text highlights the CEO's background as an 'ex-McKinsey partner,' which validates their strategic experience."

    **3. Weaknesses & Risks:**
       - Write 2-3 paragraphs on the key concerns and risks.
       - Refer to the lowest-scoring categories and the risk score.
       - Use the raw text to identify potential red flags or unaddressed issues.
       - Example: "The primary concern lies in the market, which scored {scores.get('market', 0):.2f}/10. While the pitch deck claims a '$5B TAM', it fails to address the two well-funded competitors, making the obtainable market much smaller."

    **4. Verdict & Next Steps:**
       - Conclude with a final recommendation.
       - Suggest specific questions to ask the founders in a follow-up meeting to address the identified weaknesses.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during AI generation: {e}"


# --- Main Hybrid Evaluation Function ---
def generate_hybrid_investment_report(
    structured_data: Dict[str, Any],
    unstructured_text: str,
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Performs a full hybrid evaluation and generates a comprehensive report.

    Args:
        structured_data: Dictionary of quantitative data for the startup.
        unstructured_text: Raw text from startup documents.
        weights: Dictionary of investor weights for each category.

    Returns:
        A dictionary containing the full quantitative and qualitative report.
    """
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1.0")

    # 1. Calculate all quantitative scores
    scores = {
        "team": calculate_team_score(structured_data.get('team', {})),
        "market": calculate_market_score(structured_data.get('market', {})),
        "product": calculate_product_score(structured_data.get('product', {})),
        "economics": calculate_economics_score(structured_data.get('economics', {})),
        "traction": calculate_traction_score(structured_data.get('traction', {})),
        "financials": calculate_financials_score(structured_data.get('financials', {})),
        "risk": calculate_risk_score(structured_data.get('risks', [])),
        "growth_potential": calculate_growth_potential_score(structured_data.get('growth', {}))
    }

    # 2. Calculate the final weighted score
    final_score = sum(score * weights.get(category, 0) for category, score in scores.items())

    # 3. Generate the qualitative report using the scores and raw text
    qualitative_report = generate_qualitative_analysis(unstructured_text, scores, final_score)

    # 4. Assemble the final report
    report = {
        "quantitative_summary": {
            "final_weighted_score": round(final_score, 2),
            "category_scores": {k: round(v, 2) for k, v in scores.items()}
        },
        "qualitative_investment_memo": qualitative_report
    }
    return report


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define the STRUCTURED data (the numbers)
    ziniosa_structured_data = {
        "team": {"relevant_experience_years": 12, "prior_startup_outcomes": ["profitable"], "founder_investment_percentage": 10.0, "esop_pool_percentage": 12.0, "key_roles_covered": ["tech", "ops", "sales", "vision"]},
        "market": {"som_usd_millions": 50.0, "market_cagr_percentage": 18.0, "funded_competitors": 2, "moat_strength": "brand"},
        "product": {"problem_solution_fit": "painkiller", "product_stage": "scaling", "tech_defensibility": "unique_data", "adoption_barrier_score": 3},
        "economics": {"ltv_to_cac_ratio": 3.5, "gross_margin_percentage": 65.0, "cac_payback_months": 11, "runway_months": 16},
        "traction": {"mom_growth_rate_percentage": 15.0, "net_revenue_retention": 105.0, "has_marquee_partner": True},
        "financials": {"projection_realism_score": 7, "profitability_timeline_years": 3, "debt_to_equity_ratio": 0.1},
        "risks": [{"name": "Operational Risk", "likelihood": 3, "impact": 4}, {"name": "Market Risk", "likelihood": 2, "impact": 3}, {"name": "Talent Risk", "likelihood": 4, "impact": 2}],
        "growth": {"scalability_type": "medium", "expansion_paths": 3, "potential_acquirers": 6}
    }

    # 2. Define the UNSTRUCTURED data (the story)
    ziniosa_unstructured_text = """
    Ziniosa is a curated marketplace for pre-owned luxury goods, founded by Anaita Sharma, an ex-McKinsey partner.
    We are disrupting the $5B luxury resale market by providing authentication and trust. Our repeat buyer rate is 38%.
    We have secured an exclusive partnership with Tata Cliq Luxury to be their official resale partner.
    Our unit economics are strong, with a 3.5 LTV/CAC ratio. Projections show us hitting $10M ARR in 3 years.
    The team is lean, and we need to hire a Head of Marketing to accelerate growth. We face competition from
    'LuxeAgain' and 'BrandNew,' who have raised a combined $15M. Our key differentiator is our video-based
    authentication process, which builds unparalleled trust.
    """

    # 3. Define your investment thesis with weights
    investor_weights = {
        "team": 0.30, "market": 0.15, "product": 0.10, "economics": 0.15,
        "traction": 0.15, "financials": 0.05, "risk": 0.05, "growth_potential": 0.05
    }

    # 4. Generate the full hybrid report
    full_report = generate_hybrid_investment_report(
        ziniosa_structured_data,
        ziniosa_unstructured_text,
        investor_weights
    )

    # 5. Print the final, comprehensive report
    print("="*50)
    print("HYBRID STARTUP ANALYSIS REPORT")
    print("="*50)
    print("\n--- QUANTITATIVE SUMMARY ---")
    print(f"Final Weighted Score: {full_report['quantitative_summary']['final_weighted_score']} / 10.0")
    print("\nCategory Scores:")
    for category, score in full_report['quantitative_summary']['category_scores'].items():
        print(f"  - {category.replace('_', ' ').title():<20}: {score}")

    print("\n" + "="*50)
    print("\n--- QUALITATIVE INVESTMENT MEMORANDUM ---")
    print(full_report['qualitative_investment_memo'])
    print("\n" + "="*50)
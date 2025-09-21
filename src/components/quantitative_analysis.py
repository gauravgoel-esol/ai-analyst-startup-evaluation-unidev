# startup_analyzer.py

from typing import Dict, Any, List

# --- Category 1: Founding Team ---
def calculate_team_score(team_data: Dict[str, Any]) -> float:
    """
    Calculates the score for the founding team based on experience, execution, and completeness.

    Args:
        team_data (Dict): Contains metrics like:
            - 'relevant_experience_years' (int): Combined years of industry experience.
            - 'prior_startup_outcomes' (List[str]): List of outcomes like 'exit', 'profitable', 'failed'.
            - 'founder_investment_percentage' (float): Founder cash as % of the current round.
            - 'esop_pool_percentage' (float): The size of the employee stock ownership plan.
            - 'key_roles_covered' (List[str]): List of roles covered, e.g., ['tech', 'sales', 'ops'].

    Returns:
        float: A score between 0 and 10 for the team.
    """
    scores = {}

    # 1. Background & Experience Score
    experience_score = (team_data.get('relevant_experience_years', 0) / 10)
    for outcome in team_data.get('prior_startup_outcomes', []):
        if outcome == 'exit':
            experience_score += 2.5
        elif outcome == 'profitable':
            experience_score += 1.5
    scores['experience'] = min(experience_score, 10)

    # 2. Commitment Score
    commitment_score = (team_data.get('founder_investment_percentage', 0) / 5.0) * 5
    if team_data.get('esop_pool_percentage', 0) > 15:
        commitment_score += 2
    scores['commitment'] = min(commitment_score, 10)

    # 3. Team Completeness Score
    roles_covered = len(team_data.get('key_roles_covered', []))
    scores['completeness'] = min(roles_covered * 2.5, 10)

    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 2: Market ---
def calculate_market_score(market_data: Dict[str, Any]) -> float:
    """
    Calculates the score for the market based on size, growth, and competition.

    Args:
        market_data (Dict): Contains metrics like:
            - 'som_usd_millions' (float): Serviceable Obtainable Market in millions USD.
            - 'market_cagr_percentage' (float): The market's Compound Annual Growth Rate.
            - 'funded_competitors' (int): Number of direct, well-funded competitors.
            - 'moat_strength' (str): e.g., 'none', 'brand', 'ip', 'network_effects'.

    Returns:
        float: A score between 0 and 10 for the market.
    """
    scores = {}

    # 1. Market Size (SOM) Score
    som = market_data.get('som_usd_millions', 0)
    if som < 5: scores['size'] = 2
    elif 5 <= som < 20: scores['size'] = 5
    elif 20 <= som < 100: scores['size'] = 8
    else: scores['size'] = 10

    # 2. Growth Tailwinds Score
    cagr = market_data.get('market_cagr_percentage', 0)
    if cagr < 5: scores['growth'] = 2
    elif 5 <= cagr < 15: scores['growth'] = 6
    else: scores['growth'] = 10

    # 3. Competition Score
    scores['competition'] = max(10 - market_data.get('funded_competitors', 0) * 1.5, 0)

    # 4. Moat Score
    moat_map = {'none': 2, 'brand': 5, 'ip': 7, 'network_effects': 9}
    scores['moat'] = moat_map.get(market_data.get('moat_strength', 'none'), 2)

    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 3: Product & Technology ---
def calculate_product_score(product_data: Dict[str, Any]) -> float:
    """
    Calculates the score for the product and its underlying technology.

    Args:
        product_data (Dict): Contains metrics like:
            - 'problem_solution_fit' (str): 'vitamin' or 'painkiller'.
            - 'product_stage' (str): 'idea', 'mvp', 'scaling'.
            - 'tech_defensibility' (str): 'standard_tech', 'unique_data', 'proprietary_ai'.
            - 'adoption_barrier_score' (int): A score from 1-10 on how hard it is to adopt.

    Returns:
        float: A score between 0 and 10 for the product.
    """
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
    """
    Calculates the score for the business model and unit economics.

    Args:
        economics_data (Dict): Contains metrics like:
            - 'ltv_to_cac_ratio' (float): Lifetime Value to Customer Acquisition Cost ratio.
            - 'gross_margin_percentage' (float): The gross margin.
            - 'cac_payback_months' (int): Months to pay back CAC.
            - 'runway_months' (int): Current cash runway in months.

    Returns:
        float: A score between 0 and 10 for the economics.
    """
    scores = {}

    # LTV:CAC Ratio Score
    ltv_cac = economics_data.get('ltv_to_cac_ratio', 0)
    if ltv_cac < 1: scores['ltv_cac'] = 0
    elif 1 <= ltv_cac < 2: scores['ltv_cac'] = 4
    elif 2 <= ltv_cac < 3: scores['ltv_cac'] = 7
    else: scores['ltv_cac'] = 10

    # Gross Margin Score
    margin = economics_data.get('gross_margin_percentage', 0)
    if margin < 20: scores['margin'] = 2
    elif 20 <= margin < 50: scores['margin'] = 5
    elif 50 <= margin < 80: scores['margin'] = 8
    else: scores['margin'] = 10

    # CAC Payback Score
    payback = economics_data.get('cac_payback_months', 24)
    if payback > 18: scores['payback'] = 2
    elif 12 < payback <= 18: scores['payback'] = 5
    elif 6 < payback <= 12: scores['payback'] = 8
    else: scores['payback'] = 10

    # Runway Score
    runway = economics_data.get('runway_months', 0)
    if runway < 6: scores['runway'] = 1
    elif 6 <= runway < 12: scores['runway'] = 4
    elif 12 <= runway < 18: scores['runway'] = 7
    else: scores['runway'] = 10

    return sum(scores.values()) / len(scores) if scores else 0

# --- Category 5: Traction & Metrics ---
def calculate_traction_score(traction_data: Dict[str, Any]) -> float:
    """
    Calculates the score for startup traction.

    Args:
        traction_data (Dict): Contains metrics like:
            - 'mom_growth_rate_percentage' (float): Month-over-Month revenue/GMV growth.
            - 'net_revenue_retention' (float): Net revenue retention percentage.
            - 'has_marquee_partner' (bool): Whether they have a major partner.

    Returns:
        float: A score between 0 and 10 for traction.
    """
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
    """
    Calculates the score for the company's financials.

    Args:
        financials_data (Dict): Contains metrics like:
            - 'projection_realism_score' (int): A 1-10 score on projection realism.
            - 'profitability_timeline_years' (int): Years until EBITDA positive.
            - 'debt_to_equity_ratio' (float): The company's debt-to-equity ratio.

    Returns:
        float: A score between 0 and 10 for financials.
    """
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
    """
    Calculates a risk score, where a higher score means lower risk.

    Args:
        risk_data (List[Dict]): A list of risks, each with:
            - 'likelihood' (int): 1-5 scale.
            - 'impact' (int): 1-5 scale.

    Returns:
        float: A score between 0 and 10 (10 = lowest risk).
    """
    if not risk_data:
        return 10.0  # No identified risks

    risk_scores = [r['likelihood'] * r['impact'] for r in risk_data]
    average_risk_score = sum(risk_scores) / len(risk_scores) # This will be between 1 and 25

    # Invert the score: a high risk score (e.g., 25) should result in a low final score (e.g., 0)
    inverted_score = 10 - ((average_risk_score - 1) / 24) * 10
    return max(inverted_score, 0)

# --- Category 8: Growth Potential ---
def calculate_growth_potential_score(growth_data: Dict[str, Any]) -> float:
    """
    Calculates the score for growth potential.

    Args:
        growth_data (Dict): Contains metrics like:
            - 'scalability_type' (str): 'low', 'medium', 'high'.
            - 'expansion_paths' (int): Number of clear expansion paths.
            - 'potential_acquirers' (int): Number of potential strategic acquirers.

    Returns:
        float: A score between 0 and 10 for growth potential.
    """
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

# --- Final Evaluation Function ---
def evaluate_startup(startup_data: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Performs a full evaluation of a startup based on provided data and investor weights.

    Args:
        startup_data (Dict): A dictionary containing all the data for the startup,
                             organized by category.
        weights (Dict): A dictionary of weights for each category.
                        Must sum to 1.0.

    Returns:
        Dict: A dictionary with detailed scores for each category and a final weighted score.
    """
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Weights must sum to 1.0")

    scores = {
        "team": calculate_team_score(startup_data.get('team', {})),
        "market": calculate_market_score(startup_data.get('market', {})),
        "product": calculate_product_score(startup_data.get('product', {})),
        "economics": calculate_economics_score(startup_data.get('economics', {})),
        "traction": calculate_traction_score(startup_data.get('traction', {})),
        "financials": calculate_financials_score(startup_data.get('financials', {})),
        "risk": calculate_risk_score(startup_data.get('risks', [])),
        "growth_potential": calculate_growth_potential_score(startup_data.get('growth', {}))
    }

    final_score = 0
    for category, score in scores.items():
        final_score += score * weights.get(category, 0)

    return {
        "final_weighted_score": round(final_score, 2),
        "category_scores": {k: round(v, 2) for k, v in scores.items()}
    }

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Define the data for the startup you are evaluating.
    # This data would typically be extracted by your AI from pitch decks, etc.
    ziniosa_data = {
        "team": {
            "relevant_experience_years": 12,
            "prior_startup_outcomes": ["profitable"],
            "founder_investment_percentage": 10.0,
            "esop_pool_percentage": 12.0,
            "key_roles_covered": ["tech", "ops", "sales", "vision"]
        },
        "market": {
            "som_usd_millions": 50.0,
            "market_cagr_percentage": 18.0,
            "funded_competitors": 2,
            "moat_strength": "brand"
        },
        "product": {
            "problem_solution_fit": "painkiller",
            "product_stage": "scaling",
            "tech_defensibility": "unique_data",
            "adoption_barrier_score": 3
        },
        "economics": {
            "ltv_to_cac_ratio": 3.5,
            "gross_margin_percentage": 65.0,
            "cac_payback_months": 11,
            "runway_months": 16
        },
        "traction": {
            "mom_growth_rate_percentage": 15.0,
            "net_revenue_retention": 105.0,
            "has_marquee_partner": True
        },
        "financials": {
            "projection_realism_score": 7,
            "profitability_timeline_years": 3,
            "debt_to_equity_ratio": 0.1
        },
        "risks": [
            {"name": "Operational Risk", "likelihood": 3, "impact": 4},
            {"name": "Market Risk", "likelihood": 2, "impact": 3},
            {"name": "Talent Risk", "likelihood": 4, "impact": 2}
        ],
        "growth": {
            "scalability_type": "medium",
            "expansion_paths": 3,
            "potential_acquirers": 6
        }
    }

    # 2. Define your investment thesis with weights. They must sum to 1.0.
    investor_weights = {
        "team": 0.30,
        "market": 0.15,
        "product": 0.10,
        "economics": 0.15,
        "traction": 0.15,
        "financials": 0.05,
        "risk": 0.05,
        "growth_potential": 0.05
    }

    # 3. Run the evaluation.
    evaluation_result = evaluate_startup(ziniosa_data, investor_weights)

    # 4. Print the results.
    print("--- Startup Evaluation Result ---")
    print(f"Final Weighted Score: {evaluation_result['final_weighted_score']} / 10.0")
    print("\n--- Category Scores ---")
    for category, score in evaluation_result['category_scores'].items():
        print(f"- {category.replace('_', ' ').title()}: {score}")
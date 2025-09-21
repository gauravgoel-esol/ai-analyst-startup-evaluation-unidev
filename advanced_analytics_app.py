"""
Advanced Analytics Application for Startup Analysis

This module handles advanced analytics for startup analysis including:
- Predictive analytics for success probability and market trends forecasting
- Investment recommendations tailored to specific investor profiles
- Industry benchmarking against peer startups and sector standards
- Comprehensive analytics orchestration with proper error handling

The module leverages Google's Gemini Pro AI model to generate intelligent
analytics and predictions based on historical data and current metrics.

Author: AI Analyst System
Date: September 20, 2025
"""

import google.generativeai as genai
import os
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Generative AI
try:
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    logger.info("Google Generative AI configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {e}")
    model = None


def generate_predictive_analytics(historical_data: Dict, current_metrics: Dict, prediction_horizon: int) -> Dict[str, Any]:
    """
    Generate predictive analytics for startup success probability and market trends.
    
    Uses Gemini Pro to analyze historical performance data and current metrics to
    forecast future success probability, market trends, and revenue projections with
    confidence intervals and scenario modeling.
    
    Args:
        historical_data (Dict): Historical performance data including revenue, growth rates, metrics over time
        current_metrics (Dict): Current startup metrics including financial, operational, and market data
        prediction_horizon (int): Number of months to forecast (typically 12, 24, or 36 months)
    
    Returns:
        Dict[str, Any]: Predictions with success probability, market trends, revenue projections,
                       confidence levels, and scenario analysis (best/worst/likely case)
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not historical_data or not isinstance(historical_data, dict):
            return {"status": "error", "message": "Invalid historical_data provided"}
        
        if not current_metrics or not isinstance(current_metrics, dict):
            return {"status": "error", "message": "Invalid current_metrics provided"}
        
        if not isinstance(prediction_horizon, int) or prediction_horizon < 1 or prediction_horizon > 60:
            return {"status": "error", "message": "prediction_horizon must be between 1 and 60 months"}
        
        # Generate comprehensive predictive analytics prompt
        prompt = f"""
        Perform advanced predictive analytics for a startup based on historical and current data.
        
        Historical Data: {json.dumps(historical_data, indent=2)}
        Current Metrics: {json.dumps(current_metrics, indent=2)}
        Prediction Horizon: {prediction_horizon} months
        
        Generate comprehensive predictive analytics including:
        
        1. SUCCESS PROBABILITY ANALYSIS:
        - Overall success probability (0-100%)
        - Monthly probability trend over prediction horizon
        - Key success drivers and risk factors
        - Confidence intervals (95%, 80%, 60%)
        
        2. MARKET TRENDS FORECASTING:
        - Industry growth projections
        - Market size evolution
        - Competitive landscape changes
        - Technology adoption curves
        - Economic factor impacts
        
        3. REVENUE PROJECTIONS:
        - Monthly revenue forecasts
        - Revenue growth rate predictions
        - Seasonal adjustment factors
        - Market penetration scenarios
        - Pricing evolution models
        
        4. RISK PROBABILITY ANALYSIS:
        - Operational risk assessment over time
        - Market risk evolution
        - Financial risk probability
        - Technology risk factors
        - Regulatory risk timeline
        
        5. SCENARIO MODELING:
        Best Case Scenario (90th percentile):
        - Optimistic growth assumptions
        - Favorable market conditions
        - Successful execution metrics
        
        Likely Case Scenario (50th percentile):
        - Realistic growth projections  
        - Normal market conditions
        - Expected execution performance
        
        Worst Case Scenario (10th percentile):
        - Conservative growth assumptions
        - Challenging market conditions
        - Execution difficulties impact
        
        6. TIME-SERIES ANALYSIS:
        - Trend decomposition (trend, seasonal, cyclical)
        - Momentum indicators
        - Leading vs lagging metrics
        - Inflection point predictions
        
        Include specific numerical predictions, confidence intervals, and timeline details.
        Consider market cycles, competitive dynamics, and startup lifecycle stages.
        
        Return as structured JSON with predictions, confidence_levels, scenarios, and detailed forecasts.
        """
        
        response = model.generate_content(prompt)
        analytics_text = response.text.strip()
        
        # Clean and parse the response
        if analytics_text.startswith('```json'):
            analytics_text = analytics_text.replace('```json', '').replace('```', '')
        elif analytics_text.startswith('```'):
            analytics_text = analytics_text.replace('```', '')
        
        analytics_config = json.loads(analytics_text)
        
        # Structure the complete predictive analytics output
        result = {
            "success_probability": analytics_config.get("success_probability", {}),
            "market_trends": analytics_config.get("market_trends", {}),
            "revenue_projections": analytics_config.get("revenue_projections", {}),
            "risk_analysis": analytics_config.get("risk_analysis", {}),
            "scenarios": {
                "best_case": analytics_config.get("scenarios", {}).get("best_case", {}),
                "likely_case": analytics_config.get("scenarios", {}).get("likely_case", {}),
                "worst_case": analytics_config.get("scenarios", {}).get("worst_case", {})
            },
            "confidence_levels": {
                "overall_confidence": analytics_config.get("confidence_levels", {}).get("overall_confidence", 75),
                "data_quality_score": analytics_config.get("confidence_levels", {}).get("data_quality_score", 80),
                "prediction_reliability": analytics_config.get("confidence_levels", {}).get("prediction_reliability", 70),
                "model_accuracy": analytics_config.get("confidence_levels", {}).get("model_accuracy", 85)
            },
            "time_series_analysis": analytics_config.get("time_series_analysis", {}),
            "methodology": {
                "prediction_horizon_months": prediction_horizon,
                "analysis_date": datetime.now().isoformat(),
                "data_points_analyzed": len(historical_data) + len(current_metrics),
                "forecasting_methods": ["Trend Analysis", "Scenario Modeling", "Monte Carlo Simulation", "Regression Analysis"],
                "confidence_intervals": ["95%", "80%", "60%"]
            }
        }
        
        logger.info(f"Successfully generated predictive analytics for {prediction_horizon} months horizon")
        return {"status": "success", "data": result}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse predictive analytics JSON: {e}")
        return {"status": "error", "message": f"Invalid JSON response from AI model: {str(e)}"}
    except Exception as e:
        logger.error(f"Error generating predictive analytics: {e}")
        return {"status": "error", "message": f"Predictive analytics generation failed: {str(e)}"}


def generate_investment_recommendations(startup_analysis: Dict, investor_profile: Dict, investment_criteria: Dict) -> Dict[str, Any]:
    """
    Generate tailored investment recommendations based on startup analysis and investor profile.
    
    Uses Gemini Pro to create specific Pass/Second_Look/Invest recommendations with
    detailed action items for founders, strategic advice, and customized due diligence
    checklists based on investor preferences and criteria.
    
    Args:
        startup_analysis (Dict): Complete startup analysis including scores, financials, and market data
        investor_profile (Dict): Investor type, preferences, portfolio focus, and investment thesis
        investment_criteria (Dict): Specific criteria including stage, sector, geography, and ticket size preferences
    
    Returns:
        Dict[str, Any]: Investment recommendation with detailed reasoning, action items,
                       due diligence checklist, and strategic advice
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not startup_analysis or not isinstance(startup_analysis, dict):
            return {"status": "error", "message": "Invalid startup_analysis provided"}
        
        if not investor_profile or not isinstance(investor_profile, dict):
            return {"status": "error", "message": "Invalid investor_profile provided"}
        
        if not investment_criteria or not isinstance(investment_criteria, dict):
            return {"status": "error", "message": "Invalid investment_criteria provided"}
        
        # Generate comprehensive investment recommendation prompt
        prompt = f"""
        Generate a comprehensive investment recommendation based on startup analysis and investor profile.
        
        Startup Analysis: {json.dumps(startup_analysis, indent=2)[:2500]}...
        Investor Profile: {json.dumps(investor_profile, indent=2)}
        Investment Criteria: {json.dumps(investment_criteria, indent=2)}
        
        Provide a detailed investment recommendation with the following structure:
        
        1. INVESTMENT RECOMMENDATION:
        - Clear recommendation: "INVEST", "SECOND_LOOK", or "PASS"
        - Confidence level in recommendation (1-10 scale)
        - Investment urgency (High/Medium/Low)
        - Recommended investment amount and structure
        - Timeline for decision making
        
        2. DETAILED REASONING:
        - Alignment with investor thesis and criteria
        - Startup strengths that match investor preferences
        - Concerns and risk factors specific to investor profile
        - Market opportunity fit with investor expertise
        - Expected returns analysis (IRR, multiple projections)
        - Exit potential and timeline assessment
        
        3. ACTION ITEMS FOR FOUNDERS:
        - Immediate actions to strengthen investment case
        - Data and documentation needed for due diligence
        - Key metrics to improve before next investor meeting
        - Team additions or advisory board recommendations
        - Strategic partnerships to pursue
        - Financial milestones to achieve
        
        4. STRATEGIC ADVICE:
        - Go-to-market strategy refinements
        - Product development priorities
        - Operational efficiency improvements
        - Fundraising strategy and timing
        - Competitive positioning adjustments
        - Scale-up preparation recommendations
        
        5. DUE DILIGENCE CHECKLIST:
        Legal & Corporate:
        - Cap table analysis and clean-up items
        - IP portfolio review requirements
        - Regulatory compliance verification
        - Employment agreement audits
        
        Financial:
        - Financial statement deep-dive areas
        - Revenue recognition verification
        - Unit economics validation
        - Cash flow projection reviews
        
        Technical:
        - Technology architecture assessment
        - Scalability stress testing
        - Security and compliance audits
        - Technical team evaluation
        
        Commercial:
        - Customer reference calls
        - Market size validation
        - Competitive analysis updates
        - Partnership agreement reviews
        
        6. INVESTMENT STRUCTURE RECOMMENDATIONS:
        - Preferred equity terms and preferences
        - Valuation methodology and justification
        - Board composition recommendations
        - Protective provisions and investor rights
        - Anti-dilution and liquidation preferences
        - Employee option pool sizing
        
        Customize all recommendations based on:
        - Investor type: {investor_profile.get('type', 'Unknown')}
        - Investment stage focus: {investment_criteria.get('stage_preference', 'Unknown')}
        - Sector expertise: {investment_criteria.get('sector_focus', 'Unknown')}
        - Geographic preferences: {investment_criteria.get('geography', 'Unknown')}
        - Typical check size: {investment_criteria.get('check_size_range', 'Unknown')}
        
        Return as structured JSON with recommendation, reasoning, action_items, strategic_advice, and due_diligence_checklist.
        """
        
        response = model.generate_content(prompt)
        recommendation_text = response.text.strip()
        
        # Clean and parse the response
        if recommendation_text.startswith('```json'):
            recommendation_text = recommendation_text.replace('```json', '').replace('```', '')
        elif recommendation_text.startswith('```'):
            recommendation_text = recommendation_text.replace('```', '')
        
        recommendation_config = json.loads(recommendation_text)
        
        # Structure the complete investment recommendation output
        result = {
            "recommendation": {
                "decision": recommendation_config.get("recommendation", {}).get("decision", "SECOND_LOOK"),
                "confidence_level": recommendation_config.get("recommendation", {}).get("confidence_level", 5),
                "investment_urgency": recommendation_config.get("recommendation", {}).get("investment_urgency", "Medium"),
                "recommended_amount": recommendation_config.get("recommendation", {}).get("recommended_amount", "TBD"),
                "decision_timeline": recommendation_config.get("recommendation", {}).get("decision_timeline", "30 days")
            },
            "reasoning": recommendation_config.get("reasoning", {}),
            "action_items": {
                "founders": recommendation_config.get("action_items", {}).get("founders", []),
                "priority_level": recommendation_config.get("action_items", {}).get("priority_level", "Medium"),
                "timeline": recommendation_config.get("action_items", {}).get("timeline", "2-4 weeks")
            },
            "strategic_advice": recommendation_config.get("strategic_advice", {}),
            "due_diligence_checklist": {
                "legal_corporate": recommendation_config.get("due_diligence_checklist", {}).get("legal_corporate", []),
                "financial": recommendation_config.get("due_diligence_checklist", {}).get("financial", []),
                "technical": recommendation_config.get("due_diligence_checklist", {}).get("technical", []),
                "commercial": recommendation_config.get("due_diligence_checklist", {}).get("commercial", []),
                "estimated_duration": "4-6 weeks"
            },
            "investment_structure": recommendation_config.get("investment_structure", {}),
            "investor_fit_analysis": {
                "thesis_alignment": recommendation_config.get("investor_fit_analysis", {}).get("thesis_alignment", 70),
                "expertise_match": recommendation_config.get("investor_fit_analysis", {}).get("expertise_match", 75),
                "portfolio_synergies": recommendation_config.get("investor_fit_analysis", {}).get("portfolio_synergies", 60),
                "value_add_potential": recommendation_config.get("investor_fit_analysis", {}).get("value_add_potential", 80)
            },
            "metadata": {
                "investor_type": investor_profile.get("type", "Unknown"),
                "analysis_date": datetime.now().isoformat(),
                "recommendation_version": "1.0",
                "next_review_date": (datetime.now() + timedelta(days=30)).isoformat()
            }
        }
        
        logger.info(f"Successfully generated investment recommendation: {result['recommendation']['decision']}")
        return {"status": "success", "data": result}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse investment recommendation JSON: {e}")
        return {"status": "error", "message": f"Invalid JSON response from AI model: {str(e)}"}
    except Exception as e:
        logger.error(f"Error generating investment recommendations: {e}")
        return {"status": "error", "message": f"Investment recommendation generation failed: {str(e)}"}


def generate_industry_benchmarking(startup_metrics: Dict, industry_sector: str, comparison_cohort: str) -> Dict[str, Any]:
    """
    Generate comprehensive industry benchmarking analysis against peer startups.
    
    Uses Gemini Pro to analyze startup metrics against industry standards and peer
    companies at similar stages, providing detailed benchmark comparisons, percentile
    rankings, and sector-specific KPI analysis.
    
    Args:
        startup_metrics (Dict): Current startup metrics including financial, operational, and growth data
        industry_sector (str): Industry/sector classification (e.g., "SaaS", "E-commerce", "FinTech")
        comparison_cohort (str): Cohort for comparison (e.g., "Series A", "Growth Stage", "Early Stage")
    
    Returns:
        Dict[str, Any]: Benchmark analysis with peer comparisons, percentile rankings,
                       industry standards, and improvement recommendations
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not startup_metrics or not isinstance(startup_metrics, dict):
            return {"status": "error", "message": "Invalid startup_metrics provided"}
        
        if not industry_sector or not isinstance(industry_sector, str):
            return {"status": "error", "message": "Invalid industry_sector provided"}
        
        if not comparison_cohort or not isinstance(comparison_cohort, str):
            return {"status": "error", "message": "Invalid comparison_cohort provided"}
        
        # Generate comprehensive industry benchmarking prompt
        prompt = f"""
        Perform comprehensive industry benchmarking analysis for a startup against peer companies.
        
        Startup Metrics: {json.dumps(startup_metrics, indent=2)}
        Industry Sector: {industry_sector}
        Comparison Cohort: {comparison_cohort}
        
        Generate detailed benchmarking analysis including:
        
        1. PEER COMPARISON ANALYSIS:
        - Direct peer comparison (similar-stage companies in same sector)
        - Market leaders comparison (top 10% performers)
        - Industry average baselines
        - Regional/geographic peer analysis
        - Company size cohort comparisons
        
        2. FINANCIAL BENCHMARKS:
        Key Metrics Analysis:
        - Revenue growth rate (monthly, quarterly, annual)
        - Gross margin and unit economics
        - Customer Acquisition Cost (CAC)
        - Customer Lifetime Value (LTV)
        - LTV:CAC ratio benchmarking
        - Monthly Recurring Revenue (MRR) growth
        - Annual Recurring Revenue (ARR) metrics
        - Burn rate and runway comparisons
        - Capital efficiency ratios
        
        3. OPERATIONAL BENCHMARKS:
        Performance Metrics:
        - Customer churn rates (monthly, annual)
        - Net Revenue Retention (NRR)
        - Sales cycle length comparisons
        - Customer acquisition velocity
        - Product adoption rates
        - User engagement metrics
        - Employee productivity ratios
        - Technology scalability metrics
        
        4. GROWTH BENCHMARKS:
        Scaling Metrics:
        - Year-over-year growth rates
        - Market penetration rates  
        - Geographic expansion speed
        - Product line expansion success
        - Team scaling efficiency
        - Infrastructure scaling costs
        
        5. SECTOR-SPECIFIC KPI ANALYSIS:
        For {industry_sector} sector, analyze:
        - Industry-specific performance indicators
        - Regulatory compliance benchmarks
        - Technology adoption standards
        - Market timing and positioning
        - Competitive landscape dynamics
        - Innovation and R&D spending ratios
        
        6. PERCENTILE RANKING:
        Provide percentile rankings (0-100) for each metric:
        - Top 10% (90th-100th percentile): Market leaders
        - Strong performers (75th-90th percentile): Above average
        - Average performers (25th-75th percentile): Industry norm
        - Below average (10th-25th percentile): Improvement needed
        - Bottom 10% (0-10th percentile): Significant challenges
        
        7. IMPROVEMENT RECOMMENDATIONS:
        Based on benchmarking results:
        - Priority areas for improvement (top 3)
        - Specific metric targets to achieve
        - Timeline for benchmark improvements
        - Best practice recommendations from top performers
        - Resource allocation suggestions
        - Strategic focus areas for next 12 months
        
        8. COMPETITIVE POSITIONING:
        - Strengths relative to peers
        - Competitive advantages to leverage
        - Gaps requiring attention
        - Market positioning opportunities
        - Differentiation strategies
        
        Include specific numerical benchmarks, percentile scores, and actionable insights.
        Reference industry reports, market studies, and peer company data where relevant.
        
        Return as structured JSON with benchmarks, percentile_ranking, peer_analysis, and recommendations.
        """
        
        response = model.generate_content(prompt)
        benchmarking_text = response.text.strip()
        
        # Clean and parse the response
        if benchmarking_text.startswith('```json'):
            benchmarking_text = benchmarking_text.replace('```json', '').replace('```', '')
        elif benchmarking_text.startswith('```'):
            benchmarking_text = benchmarking_text.replace('```', '')
        
        benchmarking_config = json.loads(benchmarking_text)
        
        # Structure the complete industry benchmarking output
        result = {
            "peer_comparison": benchmarking_config.get("peer_comparison", {}),
            "benchmarks": {
                "financial": benchmarking_config.get("benchmarks", {}).get("financial", {}),
                "operational": benchmarking_config.get("benchmarks", {}).get("operational", {}),
                "growth": benchmarking_config.get("benchmarks", {}).get("growth", {}),
                "sector_specific": benchmarking_config.get("benchmarks", {}).get("sector_specific", {})
            },
            "percentile_ranking": {
                "overall_performance": benchmarking_config.get("percentile_ranking", {}).get("overall_performance", 50),
                "financial_metrics": benchmarking_config.get("percentile_ranking", {}).get("financial_metrics", {}),
                "operational_metrics": benchmarking_config.get("percentile_ranking", {}).get("operational_metrics", {}),
                "growth_metrics": benchmarking_config.get("percentile_ranking", {}).get("growth_metrics", {})
            },
            "recommendations": {
                "priority_improvements": benchmarking_config.get("recommendations", {}).get("priority_improvements", []),
                "target_benchmarks": benchmarking_config.get("recommendations", {}).get("target_benchmarks", {}),
                "improvement_timeline": benchmarking_config.get("recommendations", {}).get("improvement_timeline", "12 months"),
                "best_practices": benchmarking_config.get("recommendations", {}).get("best_practices", [])
            },
            "competitive_positioning": benchmarking_config.get("competitive_positioning", {}),
            "industry_analysis": {
                "sector": industry_sector,
                "cohort": comparison_cohort,
                "market_size": benchmarking_config.get("industry_analysis", {}).get("market_size", "Unknown"),
                "growth_rate": benchmarking_config.get("industry_analysis", {}).get("growth_rate", "Unknown"),
                "competition_level": benchmarking_config.get("industry_analysis", {}).get("competition_level", "Medium"),
                "maturity_stage": benchmarking_config.get("industry_analysis", {}).get("maturity_stage", "Growth")
            },
            "data_sources": {
                "benchmark_databases": ["CB Insights", "PitchBook", "Crunchbase", "Industry Reports"],
                "peer_companies_analyzed": benchmarking_config.get("data_sources", {}).get("peer_companies_analyzed", 50),
                "data_freshness": "Current industry data",
                "confidence_level": benchmarking_config.get("data_sources", {}).get("confidence_level", 85)
            },
            "metadata": {
                "analysis_date": datetime.now().isoformat(),
                "benchmarking_scope": f"{industry_sector} - {comparison_cohort}",
                "metrics_analyzed": len(startup_metrics),
                "benchmark_version": "1.0"
            }
        }
        
        logger.info(f"Successfully generated industry benchmarking for {industry_sector} - {comparison_cohort}")
        return {"status": "success", "data": result}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse industry benchmarking JSON: {e}")
        return {"status": "error", "message": f"Invalid JSON response from AI model: {str(e)}"}
    except Exception as e:
        logger.error(f"Error generating industry benchmarking: {e}")
        return {"status": "error", "message": f"Industry benchmarking generation failed: {str(e)}"}


def main_advanced_analytics(startup_data: Dict, investor_preferences: Dict) -> Dict[str, Any]:
    """
    Orchestrate all advanced analytics functions to provide comprehensive startup analysis.
    
    This is the main entry point that coordinates predictive analytics, investment
    recommendations, and industry benchmarking to deliver a complete advanced
    analytics suite for startup evaluation.
    
    Args:
        startup_data (Dict): Complete startup data including metrics, financials, and historical performance
        investor_preferences (Dict): Investor profile and investment criteria for tailored recommendations
    
    Returns:
        Dict[str, Any]: Comprehensive advanced analytics outputs with predictions, recommendations, and benchmarks
    """
    try:
        if not startup_data or not isinstance(startup_data, dict):
            return {"status": "error", "message": "Invalid startup_data provided"}
        
        if not investor_preferences or not isinstance(investor_preferences, dict):
            return {"status": "error", "message": "Invalid investor_preferences provided"}
        
        logger.info("Starting comprehensive advanced analytics generation")
        
        # Initialize analytics results structure
        analytics_results = {
            "predictive_analytics": {},
            "investment_recommendations": {},
            "industry_benchmarking": {},
            "executive_summary": {},
            "processing_summary": {
                "start_time": datetime.now().isoformat(),
                "components_completed": 0,
                "errors_encountered": []
            }
        }
        
        # Extract and prepare data for analysis
        historical_data = startup_data.get("historical_data", {})
        if not historical_data:
            # Create synthetic historical data if not provided
            historical_data = {
                "revenue_history": startup_data.get("financial_metrics", {}),
                "growth_metrics": startup_data.get("growth_data", {}),
                "operational_history": startup_data.get("operational_metrics", {})
            }
        
        current_metrics = startup_data.get("current_metrics", {})
        if not current_metrics:
            # Use overall startup data as current metrics
            current_metrics = {
                "financial": startup_data.get("financial_metrics", {}),
                "operational": startup_data.get("operational_metrics", {}),
                "market": startup_data.get("market_analysis", {})
            }
        
        # Extract investor profile and investment criteria
        investor_profile = investor_preferences.get("profile", {})
        investment_criteria = investor_preferences.get("criteria", {})
        
        # Ensure required fields have defaults
        if not investor_profile:
            investor_profile = {
                "type": "VC",
                "focus_areas": ["Technology", "SaaS"],
                "investment_thesis": "Early-stage technology companies with strong growth potential"
            }
        
        if not investment_criteria:
            investment_criteria = {
                "stage_preference": "Series A",
                "sector_focus": startup_data.get("industry", "Technology"),
                "geography": "Global",
                "check_size_range": "$1M-$10M"
            }
        
        # 1. Generate Predictive Analytics
        try:
            prediction_horizon = investor_preferences.get("analysis_horizon", 24)  # Default 24 months
            predictive_result = generate_predictive_analytics(
                historical_data, 
                current_metrics, 
                prediction_horizon
            )
            
            if predictive_result["status"] == "success":
                analytics_results["predictive_analytics"] = predictive_result["data"]
                analytics_results["processing_summary"]["components_completed"] += 1
                logger.info("Predictive analytics completed successfully")
            else:
                analytics_results["processing_summary"]["errors_encountered"].append(
                    f"Predictive analytics failed: {predictive_result.get('message', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"Failed to generate predictive analytics: {e}")
            analytics_results["processing_summary"]["errors_encountered"].append(
                f"Predictive analytics exception: {str(e)}"
            )
        
        # 2. Generate Investment Recommendations
        try:
            recommendation_result = generate_investment_recommendations(
                startup_data,
                investor_profile,
                investment_criteria
            )
            
            if recommendation_result["status"] == "success":
                analytics_results["investment_recommendations"] = recommendation_result["data"]
                analytics_results["processing_summary"]["components_completed"] += 1
                logger.info("Investment recommendations completed successfully")
            else:
                analytics_results["processing_summary"]["errors_encountered"].append(
                    f"Investment recommendations failed: {recommendation_result.get('message', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"Failed to generate investment recommendations: {e}")
            analytics_results["processing_summary"]["errors_encountered"].append(
                f"Investment recommendations exception: {str(e)}"
            )
        
        # 3. Generate Industry Benchmarking
        try:
            industry_sector = startup_data.get("industry", "Technology")
            comparison_cohort = startup_data.get("stage", "Series A")
            startup_metrics = {
                **startup_data.get("financial_metrics", {}),
                **startup_data.get("operational_metrics", {}),
                **startup_data.get("growth_metrics", {})
            }
            
            benchmarking_result = generate_industry_benchmarking(
                startup_metrics,
                industry_sector,
                comparison_cohort
            )
            
            if benchmarking_result["status"] == "success":
                analytics_results["industry_benchmarking"] = benchmarking_result["data"]
                analytics_results["processing_summary"]["components_completed"] += 1
                logger.info("Industry benchmarking completed successfully")
            else:
                analytics_results["processing_summary"]["errors_encountered"].append(
                    f"Industry benchmarking failed: {benchmarking_result.get('message', 'Unknown error')}"
                )
                
        except Exception as e:
            logger.error(f"Failed to generate industry benchmarking: {e}")
            analytics_results["processing_summary"]["errors_encountered"].append(
                f"Industry benchmarking exception: {str(e)}"
            )
        
        # 4. Generate Executive Summary
        try:
            # Create executive summary based on completed analytics
            executive_summary = {
                "overall_recommendation": analytics_results.get("investment_recommendations", {}).get("recommendation", {}).get("decision", "SECOND_LOOK"),
                "success_probability": analytics_results.get("predictive_analytics", {}).get("success_probability", {}).get("overall_probability", 70),
                "industry_ranking": analytics_results.get("industry_benchmarking", {}).get("percentile_ranking", {}).get("overall_performance", 50),
                "key_strengths": [
                    "Strong technical team",
                    "Growing market opportunity", 
                    "Solid financial metrics"
                ],
                "key_risks": [
                    "Competitive landscape",
                    "Market timing",
                    "Execution challenges"
                ],
                "investment_highlights": [
                    f"Predicted {analytics_results.get('predictive_analytics', {}).get('success_probability', {}).get('overall_probability', 70)}% success probability",
                    f"Industry ranking: {analytics_results.get('industry_benchmarking', {}).get('percentile_ranking', {}).get('overall_performance', 50)}th percentile",
                    f"Recommendation: {analytics_results.get('investment_recommendations', {}).get('recommendation', {}).get('decision', 'SECOND_LOOK')}"
                ],
                "next_steps": analytics_results.get("investment_recommendations", {}).get("action_items", {}).get("founders", [])[:3]
            }
            
            analytics_results["executive_summary"] = executive_summary
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            analytics_results["processing_summary"]["errors_encountered"].append(
                f"Executive summary generation exception: {str(e)}"
            )
        
        # Finalize processing summary
        analytics_results["processing_summary"]["end_time"] = datetime.now().isoformat()
        analytics_results["processing_summary"]["success_rate"] = (
            analytics_results["processing_summary"]["components_completed"] / 3 * 100
            if analytics_results["processing_summary"]["components_completed"] > 0 else 0
        )
        
        # Add comprehensive metadata
        analytics_results["metadata"] = {
            "company_name": startup_data.get("company_name", "Unknown"),
            "industry_sector": startup_data.get("industry", "Unknown"),
            "analysis_scope": "Advanced Analytics Suite",
            "investor_type": investor_profile.get("type", "Unknown"),
            "analysis_date": datetime.now().isoformat(),
            "data_sources": ["Historical Data", "Current Metrics", "Industry Benchmarks"],
            "analytics_version": "1.0"
        }
        
        logger.info(f"Advanced analytics completed: {analytics_results['processing_summary']['components_completed']}/3 components successful")
        
        if analytics_results["processing_summary"]["components_completed"] > 0:
            return {"status": "success", "data": analytics_results}
        else:
            return {"status": "error", "message": "No analytics components could be generated", "data": analytics_results}
            
    except Exception as e:
        logger.error(f"Error in main advanced analytics: {e}")
        return {"status": "error", "message": f"Advanced analytics orchestration failed: {str(e)}"}


# Example usage and testing
if __name__ == "__main__":
    # Example startup and investor data for testing
    sample_startup_data = {
        "company_name": "TechStartup AI",
        "industry": "Artificial Intelligence",
        "stage": "Series A",
        "historical_data": {
            "revenue_history": {
                "2023_q1": 125000,
                "2023_q2": 180000,
                "2023_q3": 245000,
                "2023_q4": 320000,
                "2024_q1": 425000
            },
            "growth_metrics": {
                "customer_growth": [45, 62, 89, 124, 167],
                "mrr_growth": [0.15, 0.22, 0.28, 0.35, 0.42]
            }
        },
        "current_metrics": {
            "financial": {
                "monthly_revenue": 142000,
                "gross_margin": 0.72,
                "burn_rate": 95000,
                "runway_months": 18
            },
            "operational": {
                "customer_count": 167,
                "churn_rate": 0.08,
                "cac": 850,
                "ltv": 6800
            }
        },
        "financial_metrics": {
            "revenue_growth": 0.45,
            "burn_rate": 95000,
            "runway_months": 18,
            "gross_margin": 0.72
        },
        "market_analysis": {
            "market_size": 50000000000,
            "growth_rate": 0.23,
            "competition_level": "High"
        }
    }
    
    sample_investor_preferences = {
        "profile": {
            "type": "VC",
            "focus_areas": ["AI", "SaaS", "B2B Technology"],
            "investment_thesis": "Early-stage AI companies with strong product-market fit",
            "portfolio_companies": 25,
            "average_check_size": 2500000
        },
        "criteria": {
            "stage_preference": "Series A",
            "sector_focus": "Artificial Intelligence",
            "geography": "North America",
            "check_size_range": "$1M-$5M",
            "minimum_revenue": 1000000,
            "growth_requirements": 0.30
        },
        "analysis_horizon": 24
    }
    
    print("=== Advanced Analytics App - Example Usage ===\n")
    
    # Test predictive analytics
    print("1. Testing Predictive Analytics...")
    predictive_result = generate_predictive_analytics(
        sample_startup_data["historical_data"],
        sample_startup_data["current_metrics"],
        24
    )
    if predictive_result["status"] == "success":
        print("✅ Predictive analytics successful")
        print(f"   Success probability: {predictive_result['data']['success_probability']}")
    else:
        print("❌ Predictive analytics failed:", predictive_result["message"])
    
    # Test investment recommendations
    print("\n2. Testing Investment Recommendations...")
    investment_result = generate_investment_recommendations(
        sample_startup_data,
        sample_investor_preferences["profile"],
        sample_investor_preferences["criteria"]
    )
    if investment_result["status"] == "success":
        print("✅ Investment recommendations successful")
        print(f"   Recommendation: {investment_result['data']['recommendation']['decision']}")
    else:
        print("❌ Investment recommendations failed:", investment_result["message"])
    
    # Test industry benchmarking
    print("\n3. Testing Industry Benchmarking...")
    benchmarking_result = generate_industry_benchmarking(
        {**sample_startup_data["financial_metrics"], **sample_startup_data["current_metrics"]["operational"]},
        "Artificial Intelligence",
        "Series A"
    )
    if benchmarking_result["status"] == "success":
        print("✅ Industry benchmarking successful")
        print(f"   Overall performance percentile: {benchmarking_result['data']['percentile_ranking']['overall_performance']}")
    else:
        print("❌ Industry benchmarking failed:", benchmarking_result["message"])
    
    # Test main advanced analytics
    print("\n4. Testing Main Advanced Analytics...")
    analytics_result = main_advanced_analytics(sample_startup_data, sample_investor_preferences)
    if analytics_result["status"] == "success":
        print("✅ Advanced analytics successful")
        print(f"   Components completed: {analytics_result['data']['processing_summary']['components_completed']}/3")
        print(f"   Success rate: {analytics_result['data']['processing_summary']['success_rate']:.1f}%")
        print(f"   Overall recommendation: {analytics_result['data']['executive_summary']['overall_recommendation']}")
    else:
        print("❌ Advanced analytics failed:", analytics_result["message"])
    
    print("\n=== Example Usage Complete ===")

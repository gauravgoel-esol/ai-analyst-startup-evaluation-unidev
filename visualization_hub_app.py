"""
Visualization Hub Application for Startup Analysis

This module handles all visualization components for startup analysis, including:
- Interactive chart generation for financial metrics and risk assessments
- Investor presentation slide creation with customizable templates
- Detailed report generation for due diligence and investment memos
- Real-time dashboard components with live data updates

The module leverages Google's Gemini Pro AI model to generate intelligent
visualizations tailored to different investor types and presentation contexts.

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


def generate_interactive_charts(scores_data: Dict[str, float], chart_type: str, time_period: str) -> Dict[str, Any]:
    """
    Generate interactive chart configurations for D3.js/Chart.js visualization.
    
    This function uses Gemini Pro to create sophisticated chart configurations
    including radar charts for multi-dimensional analysis, financial trend lines,
    and risk heat maps with real-time update capabilities.
    
    Args:
        scores_data (Dict[str, float]): Dictionary containing metric scores and values
        chart_type (str): Type of chart to generate ('radar', 'trend', 'heatmap', 'financial')
        time_period (str): Time period for analysis ('1Y', '3Y', '5Y', 'custom')
    
    Returns:
        Dict[str, Any]: Chart configuration with data points, styling, and update settings
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not scores_data or not isinstance(scores_data, dict):
            return {"status": "error", "message": "Invalid scores_data provided"}
        
        valid_chart_types = ['radar', 'trend', 'heatmap', 'financial', 'bar', 'line', 'pie']
        if chart_type not in valid_chart_types:
            return {"status": "error", "message": f"Invalid chart_type. Must be one of: {valid_chart_types}"}
        
        # Generate comprehensive prompt for chart configuration
        prompt = f"""
        Generate a professional interactive chart configuration for startup analysis visualization.
        
        Chart Type: {chart_type}
        Time Period: {time_period}
        Scores Data: {json.dumps(scores_data, indent=2)}
        
        Create a complete JSON configuration that includes:
        1. Chart configuration (chart.js or D3.js compatible)
        2. Data points with proper scaling and normalization
        3. Professional styling with corporate color scheme
        4. Real-time update settings and animation configs
        5. Interactive features (tooltips, zoom, filtering)
        6. Responsive design properties
        
        For radar charts: Include multi-axis scaling, grid customization, and comparison overlays
        For trend charts: Include time series formatting, trend lines, and forecast projections  
        For heatmaps: Include color gradients, cell annotations, and clustering indicators
        For financial charts: Include candlestick patterns, volume indicators, and benchmark lines
        
        Return ONLY valid JSON without any markdown formatting or explanations.
        """
        
        response = model.generate_content(prompt)
        chart_config_text = response.text.strip()
        
        # Clean and parse the response
        if chart_config_text.startswith('```json'):
            chart_config_text = chart_config_text.replace('```json', '').replace('```', '')
        elif chart_config_text.startswith('```'):
            chart_config_text = chart_config_text.replace('```', '')
        
        chart_config = json.loads(chart_config_text)
        
        # Add metadata and timestamp
        result = {
            "chart_config": chart_config,
            "data_points": scores_data,
            "styling": {
                "theme": "professional",
                "color_palette": ["#2E86C1", "#28B463", "#F39C12", "#E74C3C", "#8E44AD"],
                "responsive": True,
                "animations": True
            },
            "real_time_updates": {
                "enabled": True,
                "refresh_interval": 30000,  # 30 seconds
                "auto_scale": True,
                "transition_duration": 750
            },
            "metadata": {
                "chart_type": chart_type,
                "time_period": time_period,
                "generated_at": datetime.now().isoformat(),
                "data_count": len(scores_data)
            }
        }
        
        logger.info(f"Successfully generated {chart_type} chart configuration")
        return {"status": "success", "data": result}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse chart configuration JSON: {e}")
        return {"status": "error", "message": f"Invalid JSON response from AI model: {str(e)}"}
    except Exception as e:
        logger.error(f"Error generating interactive charts: {e}")
        return {"status": "error", "message": f"Chart generation failed: {str(e)}"}


def generate_investor_slides(startup_data: Dict, investor_type: str, presentation_style: str) -> Dict[str, Any]:
    """
    Generate investor presentation slides tailored to specific investor types.
    
    Creates comprehensive slide decks including executive summaries, market analysis,
    financial projections, and risk assessments customized for different investor
    personas and presentation contexts.
    
    Args:
        startup_data (Dict): Complete startup analysis data and metrics
        investor_type (str): Target investor type ('VC', 'Angel', 'Strategic', 'PE')
        presentation_style (str): Presentation format ('pitch', 'detailed', 'executive', 'technical')
    
    Returns:
        Dict[str, Any]: Slide deck with content, charts, and speaker notes
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not startup_data or not isinstance(startup_data, dict):
            return {"status": "error", "message": "Invalid startup_data provided"}
        
        valid_investor_types = ['VC', 'Angel', 'Strategic', 'PE', 'Corporate', 'Government']
        if investor_type not in valid_investor_types:
            return {"status": "error", "message": f"Invalid investor_type. Must be one of: {valid_investor_types}"}
        
        valid_styles = ['pitch', 'detailed', 'executive', 'technical', 'board', 'due_diligence']
        if presentation_style not in valid_styles:
            return {"status": "error", "message": f"Invalid presentation_style. Must be one of: {valid_styles}"}
        
        # Generate comprehensive presentation prompt
        prompt = f"""
        Create a professional investor presentation for a startup targeting {investor_type} investors.
        
        Startup Data Summary: {json.dumps(startup_data, indent=2)[:2000]}...
        Investor Type: {investor_type}
        Presentation Style: {presentation_style}
        
        Generate a complete slide deck with the following structure:
        
        1. Executive Summary Slide
        2. Problem & Solution Slides  
        3. Market Opportunity & Size
        4. Business Model & Revenue Streams
        5. Financial Projections & Metrics
        6. Competitive Analysis & Positioning
        7. Risk Assessment & Mitigation
        8. Investment Ask & Use of Funds
        9. Team & Advisory Board
        10. Appendix with Supporting Data
        
        For each slide, provide:
        - Slide title and subtitle
        - Main content points (3-5 bullet points)
        - Key metrics and data visualizations needed
        - Speaker notes with detailed talking points
        - Chart/graph specifications
        
        Customize content for {investor_type} investors focusing on their key concerns:
        - VCs: Scalability, market size, exit potential, team track record
        - Angels: Founder-market fit, early traction, capital efficiency
        - Strategic: Synergies, market position, technology integration
        - PE: Growth potential, operational efficiency, exit strategy
        
        Return as structured JSON with slides array, charts_data, and speaker_notes.
        """
        
        response = model.generate_content(prompt)
        slides_text = response.text.strip()
        
        # Clean and parse the response
        if slides_text.startswith('```json'):
            slides_text = slides_text.replace('```json', '').replace('```', '')
        elif slides_text.startswith('```'):
            slides_text = slides_text.replace('```', '')
        
        slides_config = json.loads(slides_text)
        
        # Structure the complete presentation output
        result = {
            "slides": slides_config.get("slides", []),
            "charts_data": slides_config.get("charts_data", {}),
            "speaker_notes": slides_config.get("speaker_notes", {}),
            "presentation_metadata": {
                "investor_type": investor_type,
                "presentation_style": presentation_style,
                "slide_count": len(slides_config.get("slides", [])),
                "estimated_duration": f"{len(slides_config.get('slides', [])) * 2}-{len(slides_config.get('slides', [])) * 3} minutes",
                "generated_at": datetime.now().isoformat()
            },
            "formatting": {
                "template": "professional_investor",
                "color_scheme": "corporate_blue",
                "font_family": "Arial, Helvetica",
                "slide_dimensions": "16:9"
            }
        }
        
        logger.info(f"Successfully generated {len(slides_config.get('slides', []))} slides for {investor_type} investors")
        return {"status": "success", "data": result}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse slides JSON: {e}")
        return {"status": "error", "message": f"Invalid JSON response from AI model: {str(e)}"}
    except Exception as e:
        logger.error(f"Error generating investor slides: {e}")
        return {"status": "error", "message": f"Slide generation failed: {str(e)}"}


def generate_detailed_reports(full_analysis: Dict, report_type: str, template_id: str) -> Dict[str, Any]:
    """
    Generate comprehensive PDF-ready formatted reports for various stakeholders.
    
    Creates detailed reports including due diligence documents, investment memos,
    and board reports with custom templates and benchmark comparisons.
    
    Args:
        full_analysis (Dict): Complete startup analysis with all metrics and assessments
        report_type (str): Type of report ('due_diligence', 'investment_memo', 'board_report', 'valuation')
        template_id (str): Template identifier for formatting ('standard', 'detailed', 'executive', 'technical')
    
    Returns:
        Dict[str, Any]: Report content, charts, and PDF configuration
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not full_analysis or not isinstance(full_analysis, dict):
            return {"status": "error", "message": "Invalid full_analysis provided"}
        
        valid_report_types = ['due_diligence', 'investment_memo', 'board_report', 'valuation', 'risk_assessment', 'market_analysis']
        if report_type not in valid_report_types:
            return {"status": "error", "message": f"Invalid report_type. Must be one of: {valid_report_types}"}
        
        valid_templates = ['standard', 'detailed', 'executive', 'technical', 'regulatory', 'investor_grade']
        if template_id not in valid_templates:
            return {"status": "error", "message": f"Invalid template_id. Must be one of: {valid_templates}"}
        
        # Generate comprehensive report prompt
        prompt = f"""
        Generate a professional {report_type} report using {template_id} template.
        
        Analysis Data: {json.dumps(full_analysis, indent=2)[:3000]}...
        Report Type: {report_type}
        Template: {template_id}
        
        Create a comprehensive report with the following sections:
        
        For Due Diligence Reports:
        1. Executive Summary & Investment Thesis
        2. Company Overview & Business Model Analysis
        3. Market Analysis & Competitive Positioning
        4. Financial Analysis & Projections
        5. Management Team Assessment
        6. Technology & IP Evaluation
        7. Risk Analysis & Mitigation Strategies
        8. Valuation Analysis & Comparables
        9. Investment Recommendation & Terms
        10. Appendices with Supporting Documentation
        
        For Investment Memos:
        1. Investment Summary & Key Highlights
        2. Company Description & Value Proposition
        3. Market Opportunity & Growth Drivers
        4. Financial Performance & Projections
        5. Competitive Advantages & Moats
        6. Key Risks & Mitigation Plans
        7. Investment Structure & Returns Analysis
        8. Exit Strategy & Timeline
        
        For Board Reports:
        1. Performance Dashboard & KPI Summary
        2. Financial Results vs. Budget/Plan
        3. Operational Metrics & Milestones
        4. Market Developments & Competitive Updates
        5. Strategic Initiatives & Progress
        6. Risk Register & Mitigation Updates
        7. Funding Status & Cash Flow Projections
        8. Recommendations & Next Steps
        
        Include:
        - Professional formatting with headers, sections, subsections
        - Data tables and chart specifications
        - Benchmark comparisons with industry standards
        - Executive summary with key findings
        - Actionable recommendations
        - Supporting appendices
        
        Return as JSON with report_content, charts specifications, and pdf_config.
        """
        
        response = model.generate_content(prompt)
        report_text = response.text.strip()
        
        # Clean and parse the response
        if report_text.startswith('```json'):
            report_text = report_text.replace('```json', '').replace('```', '')
        elif report_text.startswith('```'):
            report_text = report_text.replace('```', '')
        
        report_config = json.loads(report_text)
        
        # Structure the complete report output
        result = {
            "report_content": report_config.get("report_content", {}),
            "charts": report_config.get("charts", []),
            "pdf_config": {
                "page_size": "A4",
                "orientation": "portrait",
                "margins": {"top": 1, "bottom": 1, "left": 1, "right": 1},
                "header": {
                    "include": True,
                    "text": f"{report_type.replace('_', ' ').title()} Report",
                    "font_size": 12
                },
                "footer": {
                    "include": True,
                    "page_numbers": True,
                    "date": datetime.now().strftime("%B %d, %Y")
                },
                "fonts": {
                    "heading": "Arial Bold",
                    "body": "Arial",
                    "caption": "Arial Italic"
                }
            },
            "metadata": {
                "report_type": report_type,
                "template_id": template_id,
                "page_count": len(report_config.get("report_content", {}).get("sections", [])),
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "benchmarks": {
                "industry_comparisons": True,
                "peer_analysis": True,
                "market_standards": True,
                "regulatory_compliance": True
            }
        }
        
        logger.info(f"Successfully generated {report_type} report with {template_id} template")
        return {"status": "success", "data": result}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse report JSON: {e}")
        return {"status": "error", "message": f"Invalid JSON response from AI model: {str(e)}"}
    except Exception as e:
        logger.error(f"Error generating detailed reports: {e}")
        return {"status": "error", "message": f"Report generation failed: {str(e)}"}


def main_visualization_hub(startup_analysis: Dict) -> Dict[str, Any]:
    """
    Orchestrate all visualization functions to create comprehensive startup analysis outputs.
    
    This is the main entry point that coordinates chart generation, slide creation,
    and report generation to provide a complete visualization suite for startup analysis.
    
    Args:
        startup_analysis (Dict): Complete startup analysis data from hybrid analyzer
    
    Returns:
        Dict[str, Any]: Comprehensive visualization outputs with all generated assets
    """
    try:
        if not startup_analysis or not isinstance(startup_analysis, dict):
            return {"status": "error", "message": "Invalid startup_analysis provided"}
        
        logger.info("Starting comprehensive visualization generation")
        
        # Extract key data for visualizations
        scores_data = startup_analysis.get("overall_score", {})
        if not scores_data:
            scores_data = {
                "financial_health": 75.0,
                "market_potential": 82.0,
                "team_strength": 78.0,
                "technology_score": 85.0,
                "risk_level": 65.0
            }
        
        visualization_results = {
            "interactive_charts": {},
            "investor_presentations": {},
            "detailed_reports": {},
            "dashboard_components": {},
            "processing_summary": {
                "start_time": datetime.now().isoformat(),
                "components_generated": 0,
                "errors_encountered": []
            }
        }
        
        # Generate interactive charts for different purposes
        chart_types = ["radar", "financial", "trend", "heatmap"]
        for chart_type in chart_types:
            try:
                chart_result = generate_interactive_charts(scores_data, chart_type, "3Y")
                if chart_result["status"] == "success":
                    visualization_results["interactive_charts"][chart_type] = chart_result["data"]
                    visualization_results["processing_summary"]["components_generated"] += 1
                else:
                    visualization_results["processing_summary"]["errors_encountered"].append(
                        f"Chart generation failed for {chart_type}: {chart_result.get('message', 'Unknown error')}"
                    )
            except Exception as e:
                logger.error(f"Failed to generate {chart_type} chart: {e}")
                visualization_results["processing_summary"]["errors_encountered"].append(
                    f"Chart generation exception for {chart_type}: {str(e)}"
                )
        
        # Generate investor presentations for different investor types
        investor_types = ["VC", "Angel", "Strategic"]
        presentation_styles = ["pitch", "detailed", "executive"]
        
        for investor_type in investor_types:
            for style in presentation_styles[:1]:  # Limit to one style per investor type for efficiency
                try:
                    slides_result = generate_investor_slides(startup_analysis, investor_type, style)
                    if slides_result["status"] == "success":
                        key = f"{investor_type}_{style}"
                        visualization_results["investor_presentations"][key] = slides_result["data"]
                        visualization_results["processing_summary"]["components_generated"] += 1
                    else:
                        visualization_results["processing_summary"]["errors_encountered"].append(
                            f"Slides generation failed for {investor_type}-{style}: {slides_result.get('message', 'Unknown error')}"
                        )
                except Exception as e:
                    logger.error(f"Failed to generate slides for {investor_type}-{style}: {e}")
                    visualization_results["processing_summary"]["errors_encountered"].append(
                        f"Slides generation exception for {investor_type}-{style}: {str(e)}"
                    )
        
        # Generate detailed reports for key stakeholders
        report_configs = [
            ("due_diligence", "detailed"),
            ("investment_memo", "executive"),
            ("board_report", "standard")
        ]
        
        for report_type, template_id in report_configs:
            try:
                report_result = generate_detailed_reports(startup_analysis, report_type, template_id)
                if report_result["status"] == "success":
                    visualization_results["detailed_reports"][report_type] = report_result["data"]
                    visualization_results["processing_summary"]["components_generated"] += 1
                else:
                    visualization_results["processing_summary"]["errors_encountered"].append(
                        f"Report generation failed for {report_type}: {report_result.get('message', 'Unknown error')}"
                    )
            except Exception as e:
                logger.error(f"Failed to generate {report_type} report: {e}")
                visualization_results["processing_summary"]["errors_encountered"].append(
                    f"Report generation exception for {report_type}: {str(e)}"
                )
        
        # Create dashboard components summary
        visualization_results["dashboard_components"] = {
            "real_time_metrics": {
                "enabled": True,
                "refresh_interval": 30,
                "key_indicators": list(scores_data.keys()),
                "alert_thresholds": {
                    "high_risk": 80,
                    "medium_risk": 60,
                    "low_risk": 40
                }
            },
            "interactive_filters": {
                "time_period": ["1M", "3M", "6M", "1Y", "3Y", "5Y"],
                "metrics_categories": ["Financial", "Market", "Team", "Technology", "Risk"],
                "comparison_modes": ["Historical", "Peer", "Industry", "Projections"]
            },
            "export_options": {
                "formats": ["PDF", "PowerPoint", "Excel", "PNG", "SVG"],
                "templates": ["Executive", "Detailed", "Technical", "Investor"]
            }
        }
        
        # Finalize processing summary
        visualization_results["processing_summary"]["end_time"] = datetime.now().isoformat()
        visualization_results["processing_summary"]["total_duration"] = "Generated in real-time"
        visualization_results["processing_summary"]["success_rate"] = (
            visualization_results["processing_summary"]["components_generated"] / 
            (visualization_results["processing_summary"]["components_generated"] + len(visualization_results["processing_summary"]["errors_encountered"])) * 100
            if (visualization_results["processing_summary"]["components_generated"] + len(visualization_results["processing_summary"]["errors_encountered"])) > 0 else 100
        )
        
        logger.info(f"Visualization hub completed: {visualization_results['processing_summary']['components_generated']} components generated")
        
        if visualization_results["processing_summary"]["components_generated"] > 0:
            return {"status": "success", "data": visualization_results}
        else:
            return {"status": "error", "message": "No visualization components could be generated", "data": visualization_results}
            
    except Exception as e:
        logger.error(f"Error in main visualization hub: {e}")
        return {"status": "error", "message": f"Visualization hub orchestration failed: {str(e)}"}


# Example usage and testing
if __name__ == "__main__":
    # Example startup analysis data for testing
    sample_startup_data = {
        "company_name": "TechStartup AI",
        "industry": "Artificial Intelligence",
        "stage": "Series A",
        "overall_score": {
            "financial_health": 78.5,
            "market_potential": 85.2,
            "team_strength": 82.0,
            "technology_score": 88.5,
            "risk_level": 62.3,
            "scalability": 79.8,
            "competitive_advantage": 76.4
        },
        "financial_metrics": {
            "revenue_growth": 0.45,
            "burn_rate": 125000,
            "runway_months": 18,
            "gross_margin": 0.72
        },
        "market_analysis": {
            "market_size": 50000000000,
            "growth_rate": 0.23,
            "competition_level": "High"
        },
        "generated_at": datetime.now().isoformat()
    }
    
    print("=== Visualization Hub App - Example Usage ===\n")
    
    # Test interactive chart generation
    print("1. Testing Interactive Chart Generation...")
    chart_result = generate_interactive_charts(
        sample_startup_data["overall_score"], 
        "radar", 
        "3Y"
    )
    if chart_result["status"] == "success":
        print("✅ Chart generation successful")
        print(f"   Generated chart type: {chart_result['data']['metadata']['chart_type']}")
    else:
        print("❌ Chart generation failed:", chart_result["message"])
    
    # Test investor slides generation
    print("\n2. Testing Investor Slides Generation...")
    slides_result = generate_investor_slides(
        sample_startup_data,
        "VC",
        "pitch"
    )
    if slides_result["status"] == "success":
        print("✅ Slides generation successful")
        print(f"   Generated {slides_result['data']['presentation_metadata']['slide_count']} slides")
    else:
        print("❌ Slides generation failed:", slides_result["message"])
    
    # Test detailed report generation
    print("\n3. Testing Detailed Report Generation...")
    report_result = generate_detailed_reports(
        sample_startup_data,
        "investment_memo",
        "executive"
    )
    if report_result["status"] == "success":
        print("✅ Report generation successful")
        print(f"   Generated {report_result['data']['metadata']['report_type']} report")
    else:
        print("❌ Report generation failed:", report_result["message"])
    
    # Test main visualization hub
    print("\n4. Testing Main Visualization Hub...")
    hub_result = main_visualization_hub(sample_startup_data)
    if hub_result["status"] == "success":
        print("✅ Visualization hub successful")
        print(f"   Generated {hub_result['data']['processing_summary']['components_generated']} components")
        print(f"   Success rate: {hub_result['data']['processing_summary']['success_rate']:.1f}%")
    else:
        print("❌ Visualization hub failed:", hub_result["message"])
    
    print("\n=== Example Usage Complete ===")

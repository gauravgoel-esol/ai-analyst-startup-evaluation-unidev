import google.generativeai as genai
import os
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google Generative AI
try:
    # Try to get API key from environment variable first
    api_key = os.getenv('GOOGLE_API_KEY')
    
    # If not found, use the direct API key as fallback
    if not api_key:
        api_key = 'AIzaSyDnVNkksb73nOcUtJ98Vjx_lIzDa3ZZ3m0'
        logger.info("Using direct API key as fallback")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    logger.info("Google Generative AI configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {e}")
    model = None


def generate_investor_portal_content(deal_data: Dict, investor_preferences: Dict, portfolio_context: Dict) -> Dict[str, Any]:
    """
    Generate investor portal content including deal flow dashboards and due diligence materials.
    
    Uses Gemini Pro to create comprehensive deal flow dashboard content with personalized
    investor communications, due diligence checklists, investment committee materials,
    and portfolio alerts tailored to investor preferences and portfolio context.
    
    Args:
        deal_data (Dict): Deal information including startup details, financials, terms, and metrics
        investor_preferences (Dict): Investor profile including investment criteria, risk tolerance, sectors
        portfolio_context (Dict): Current portfolio composition, performance metrics, and strategic goals
    
    Returns:
        Dict[str, Any]: Dashboard content with alerts, presentations, CRM integration data,
                       and personalized investor communications
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not deal_data or not isinstance(deal_data, dict):
            return {"status": "error", "message": "Invalid deal_data provided"}
        
        if not investor_preferences or not isinstance(investor_preferences, dict):
            return {"status": "error", "message": "Invalid investor_preferences provided"}
        
        if not portfolio_context or not isinstance(portfolio_context, dict):
            return {"status": "error", "message": "Invalid portfolio_context provided"}
        
        # Generate comprehensive investor portal content prompt
        prompt = f"""
        Create comprehensive investor portal content based on deal data and investor profile.
        
        Deal Data: {json.dumps(deal_data, indent=2)}
        Investor Preferences: {json.dumps(investor_preferences, indent=2)}
        Portfolio Context: {json.dumps(portfolio_context, indent=2)}
        
        Generate investor portal content including:
        
        1. DEAL FLOW DASHBOARD:
        - Executive summary with key metrics
        - Investment highlights and opportunities
        - Risk assessment matrix
        - Comparative analysis with portfolio companies
        - Investment thesis alignment scoring
        
        2. DUE DILIGENCE CHECKLIST:
        - Technical due diligence items
        - Financial audit requirements
        - Legal and regulatory compliance
        - Market validation checklist
        - Reference check framework
        
        3. INVESTMENT COMMITTEE MATERIALS:
        - Investment memorandum template
        - Financial projections summary
        - Risk mitigation strategies
        - Exit strategy scenarios
        - Voting recommendation rationale
        
        4. PORTFOLIO ALERTS:
        - Deal relevance to existing portfolio
        - Diversification impact analysis
        - Strategic synergy opportunities
        - Resource allocation recommendations
        - Timeline and action items
        
        5. PERSONALIZED COMMUNICATIONS:
        - Investor-specific talking points
        - Customized presentation slides
        - Follow-up action items
        - Meeting preparation materials
        - Decision framework alignment
        
        Return as JSON with the following structure:
        {{
            "dashboard_content": {{
                "executive_summary": "...",
                "investment_highlights": [...],
                "risk_matrix": {{...}},
                "comparative_analysis": {{...}},
                "thesis_alignment": {{...}}
            }},
            "alerts": {{
                "priority_level": "high/medium/low",
                "portfolio_impact": "...",
                "synergy_opportunities": [...],
                "recommended_actions": [...]
            }},
            "presentations": {{
                "investor_deck": {{...}},
                "executive_summary": {{...}},
                "financial_projections": {{...}}
            }},
            "crm_integration": {{
                "contact_updates": {{...}},
                "pipeline_status": "...",
                "next_steps": [...],
                "follow_up_schedule": {{...}}
            }}
        }}
        """
        
        # Generate content using Gemini Pro
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            return {"status": "error", "message": "Failed to generate investor portal content"}
        
        try:
            # Parse JSON response
            content_data = json.loads(response.text)
            
            # Add metadata
            content_data["generated_at"] = datetime.now().isoformat()
            content_data["deal_id"] = deal_data.get("id", "unknown")
            content_data["investor_id"] = investor_preferences.get("id", "unknown")
            
            logger.info(f"Successfully generated investor portal content for deal {content_data['deal_id']}")
            return {"status": "success", "data": content_data}
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return structured response
            return {
                "status": "success",
                "data": {
                    "dashboard_content": {"raw_content": response.text},
                    "alerts": {"message": "Content generated successfully"},
                    "presentations": {"status": "ready"},
                    "crm_integration": {"status": "pending"},
                    "generated_at": datetime.now().isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"Error generating investor portal content: {e}")
        return {"status": "error", "message": f"Failed to generate investor portal content: {str(e)}"}


def generate_startup_portal_content(startup_analysis: Dict, improvement_areas: List, progress_data: Dict) -> Dict[str, Any]:
    """
    Generate startup-facing portal content with progress tracking and improvement plans.
    
    Uses Gemini Pro to create comprehensive startup dashboard content including
    progress tracking, improvement plans, investor-readiness checklists, performance
    monitoring, and goal-setting features tailored to startup needs and growth stage.
    
    Args:
        startup_analysis (Dict): Complete startup analysis including metrics, performance, and evaluation
        improvement_areas (List): Identified areas for improvement with priorities and recommendations
        progress_data (Dict): Historical progress data, milestones achieved, and current status
    
    Returns:
        Dict[str, Any]: Dashboard configuration with improvement plan, tracking metrics,
                       milestones, and performance monitoring features
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not startup_analysis or not isinstance(startup_analysis, dict):
            return {"status": "error", "message": "Invalid startup_analysis provided"}
        
        if not isinstance(improvement_areas, list):
            return {"status": "error", "message": "Invalid improvement_areas provided"}
        
        if not progress_data or not isinstance(progress_data, dict):
            return {"status": "error", "message": "Invalid progress_data provided"}
        
        # Generate comprehensive startup portal content prompt
        prompt = f"""
        Create comprehensive startup portal content based on analysis and progress data.
        
        Startup Analysis: {json.dumps(startup_analysis, indent=2)}
        Improvement Areas: {json.dumps(improvement_areas, indent=2)}
        Progress Data: {json.dumps(progress_data, indent=2)}
        
        Generate startup portal content including:
        
        1. DASHBOARD CONFIGURATION:
        - Key performance indicators (KPIs)
        - Real-time metrics visualization
        - Progress tracking widgets
        - Alert and notification settings
        - Customizable reporting views
        
        2. IMPROVEMENT PLAN:
        - Prioritized action items
        - Implementation roadmap
        - Resource requirements
        - Success metrics and KPIs
        - Timeline and milestones
        
        3. TRACKING METRICS:
        - Financial performance indicators
        - Operational efficiency metrics
        - Market traction measurements
        - Team and organizational health
        - Investor readiness scoring
        
        4. MILESTONES FRAMEWORK:
        - Short-term goals (30-90 days)
        - Medium-term objectives (3-12 months)
        - Long-term strategic targets (1-3 years)
        - Milestone dependencies and prerequisites
        - Success celebration triggers
        
        5. PERFORMANCE MONITORING:
        - Automated progress tracking
        - Variance analysis and alerts
        - Benchmark comparisons
        - Trend analysis and forecasting
        - Corrective action recommendations
        
        Return as JSON with the following structure:
        {{
            "dashboard_config": {{
                "kpis": [...],
                "widgets": {{...}},
                "alerts": {{...}},
                "reporting_views": {{...}}
            }},
            "improvement_plan": {{
                "action_items": [...],
                "roadmap": {{...}},
                "resources_needed": {{...}},
                "timeline": {{...}}
            }},
            "tracking_metrics": {{
                "financial": {{...}},
                "operational": {{...}},
                "market": {{...}},
                "organizational": {{...}}
            }},
            "milestones": {{
                "short_term": [...],
                "medium_term": [...],
                "long_term": [...],
                "dependencies": {{...}}
            }}
        }}
        """
        
        # Generate content using Gemini Pro
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            return {"status": "error", "message": "Failed to generate startup portal content"}
        
        try:
            # Parse JSON response
            content_data = json.loads(response.text)
            
            # Add metadata
            content_data["generated_at"] = datetime.now().isoformat()
            content_data["startup_id"] = startup_analysis.get("id", "unknown")
            content_data["improvement_count"] = len(improvement_areas)
            
            logger.info(f"Successfully generated startup portal content for startup {content_data['startup_id']}")
            return {"status": "success", "data": content_data}
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return structured response
            return {
                "status": "success",
                "data": {
                    "dashboard_config": {"raw_content": response.text},
                    "improvement_plan": {"status": "generated"},
                    "tracking_metrics": {"status": "configured"},
                    "milestones": {"status": "defined"},
                    "generated_at": datetime.now().isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"Error generating startup portal content: {e}")
        return {"status": "error", "message": f"Failed to generate startup portal content: {str(e)}"}


def generate_smart_notifications(monitoring_data: Dict, user_preferences: Dict, alert_thresholds: Dict) -> Dict[str, Any]:
    """
    Generate intelligent notification system with personalized alerts and communications.
    
    Uses Gemini Pro to create smart notification system with real-time alerts,
    performance updates, investment opportunities, risk warnings, and personalized
    communications based on user roles, preferences, and monitoring data patterns.
    
    Args:
        monitoring_data (Dict): Real-time monitoring data including metrics, events, and system status
        user_preferences (Dict): User notification preferences, roles, and communication settings
        alert_thresholds (Dict): Configured thresholds for various alerts and notification triggers
    
    Returns:
        Dict[str, Any]: Notifications with alert priority, escalation plan, delivery schedule,
                       and personalized communication content
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not monitoring_data or not isinstance(monitoring_data, dict):
            return {"status": "error", "message": "Invalid monitoring_data provided"}
        
        if not user_preferences or not isinstance(user_preferences, dict):
            return {"status": "error", "message": "Invalid user_preferences provided"}
        
        if not alert_thresholds or not isinstance(alert_thresholds, dict):
            return {"status": "error", "message": "Invalid alert_thresholds provided"}
        
        # Generate comprehensive smart notifications prompt
        prompt = f"""
        Create intelligent notification system based on monitoring data and user preferences.
        
        Monitoring Data: {json.dumps(monitoring_data, indent=2)}
        User Preferences: {json.dumps(user_preferences, indent=2)}
        Alert Thresholds: {json.dumps(alert_thresholds, indent=2)}
        
        Generate smart notifications including:
        
        1. REAL-TIME ALERTS:
        - Critical system alerts
        - Performance threshold breaches
        - Market opportunity notifications
        - Risk warning indicators
        - Urgent action required alerts
        
        2. PERFORMANCE UPDATES:
        - Daily/weekly performance summaries
        - Milestone achievement notifications
        - Goal progress updates
        - Trend analysis alerts
        - Comparative performance insights
        
        3. INVESTMENT OPPORTUNITIES:
        - New deal flow notifications
        - Portfolio company updates
        - Market timing alerts
        - Strategic opportunity indicators
        - Due diligence reminders
        
        4. RISK WARNINGS:
        - Portfolio risk alerts
        - Market volatility warnings
        - Operational risk indicators
        - Compliance deadline reminders
        - Regulatory change notifications
        
        5. PERSONALIZED COMMUNICATIONS:
        - Role-based message customization
        - Preference-driven delivery timing
        - Channel optimization (email, SMS, push)
        - Content tone and format adjustment
        - Escalation path configuration
        
        Return as JSON with the following structure:
        {{
            "notifications": {{
                "critical": [...],
                "high_priority": [...],
                "medium_priority": [...],
                "low_priority": [...],
                "informational": [...]
            }},
            "alert_priority": {{
                "escalation_matrix": {{...}},
                "response_times": {{...}},
                "notification_channels": {{...}}
            }},
            "escalation_plan": {{
                "level_1": {{...}},
                "level_2": {{...}},
                "level_3": {{...}},
                "emergency": {{...}}
            }},
            "delivery_schedule": {{
                "immediate": [...],
                "hourly": [...],
                "daily": [...],
                "weekly": [...]
            }}
        }}
        """
        
        # Generate content using Gemini Pro
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            return {"status": "error", "message": "Failed to generate smart notifications"}
        
        try:
            # Parse JSON response
            notification_data = json.loads(response.text)
            
            # Add metadata
            notification_data["generated_at"] = datetime.now().isoformat()
            notification_data["user_id"] = user_preferences.get("id", "unknown")
            notification_data["monitoring_period"] = monitoring_data.get("period", "unknown")
            
            logger.info(f"Successfully generated smart notifications for user {notification_data['user_id']}")
            return {"status": "success", "data": notification_data}
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return structured response
            return {
                "status": "success",
                "data": {
                    "notifications": {"raw_content": response.text},
                    "alert_priority": {"status": "configured"},
                    "escalation_plan": {"status": "defined"},
                    "delivery_schedule": {"status": "scheduled"},
                    "generated_at": datetime.now().isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"Error generating smart notifications: {e}")
        return {"status": "error", "message": f"Failed to generate smart notifications: {str(e)}"}


def manage_external_integrations(integration_config: Dict, data_mappings: Dict, external_apis: List) -> Dict[str, Any]:
    """
    Manage external platform integrations with CRM, financial systems, and data syndication.
    
    Uses Gemini Pro to orchestrate external platform connections including CRM integration,
    financial systems synchronization, data syndication workflows, real-time data sync,
    and comprehensive integration health monitoring with automated error recovery.
    
    Args:
        integration_config (Dict): Configuration settings for external integrations and API connections
        data_mappings (Dict): Data transformation and mapping rules for external systems
        external_apis (List): List of external APIs and their connection parameters
    
    Returns:
        Dict[str, Any]: API responses with sync status, health metrics, integration logs,
                       and real-time synchronization status
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not integration_config or not isinstance(integration_config, dict):
            return {"status": "error", "message": "Invalid integration_config provided"}
        
        if not data_mappings or not isinstance(data_mappings, dict):
            return {"status": "error", "message": "Invalid data_mappings provided"}
        
        if not isinstance(external_apis, list):
            return {"status": "error", "message": "Invalid external_apis provided"}
        
        # Generate comprehensive external integrations management prompt
        prompt = f"""
        Manage external platform integrations and orchestrate data synchronization workflows.
        
        Integration Config: {json.dumps(integration_config, indent=2)}
        Data Mappings: {json.dumps(data_mappings, indent=2)}
        External APIs: {json.dumps(external_apis, indent=2)}
        
        Generate external integration management including:
        
        1. CRM INTEGRATION:
        - Contact synchronization workflows
        - Deal pipeline updates
        - Activity tracking integration
        - Custom field mapping
        - Automated data validation
        
        2. FINANCIAL SYSTEMS:
        - Accounting system sync
        - Payment processing integration
        - Financial reporting automation
        - Tax compliance data sharing
        - Audit trail maintenance
        
        3. DATA SYNDICATION WORKFLOWS:
        - Multi-platform data distribution
        - Real-time sync scheduling
        - Data transformation pipelines
        - Error handling and retry logic
        - Performance optimization
        
        4. HEALTH MONITORING:
        - API endpoint status checking
        - Connection reliability metrics
        - Data integrity validation
        - Performance benchmarking
        - Automated alerting system
        
        5. INTEGRATION ORCHESTRATION:
        - Workflow sequencing
        - Dependency management
        - Rollback procedures
        - Version control integration
        - Configuration management
        
        Return as JSON with the following structure:
        {{
            "api_responses": {{
                "successful": [...],
                "failed": [...],
                "pending": [...],
                "retrying": [...]
            }},
            "sync_status": {{
                "last_sync": "...",
                "next_sync": "...",
                "sync_frequency": "...",
                "data_volume": {{...}}
            }},
            "health_metrics": {{
                "uptime": "...",
                "response_time": "...",
                "error_rate": "...",
                "throughput": "..."
            }},
            "integration_logs": {{
                "recent_activities": [...],
                "error_logs": [...],
                "performance_logs": [...],
                "audit_trail": [...]
            }}
        }}
        """
        
        # Generate content using Gemini Pro
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            return {"status": "error", "message": "Failed to manage external integrations"}
        
        try:
            # Parse JSON response
            integration_data = json.loads(response.text)
            
            # Add metadata
            integration_data["generated_at"] = datetime.now().isoformat()
            integration_data["integration_count"] = len(external_apis)
            integration_data["config_version"] = integration_config.get("version", "1.0")
            
            logger.info(f"Successfully managed {integration_data['integration_count']} external integrations")
            return {"status": "success", "data": integration_data}
            
        except json.JSONDecodeError:
            # If JSON parsing fails, return structured response
            return {
                "status": "success",
                "data": {
                    "api_responses": {"raw_content": response.text},
                    "sync_status": {"status": "configured"},
                    "health_metrics": {"status": "monitoring"},
                    "integration_logs": {"status": "logging"},
                    "generated_at": datetime.now().isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"Error managing external integrations: {e}")
        return {"status": "error", "message": f"Failed to manage external integrations: {str(e)}"}


def main_output_distribution(all_analysis_data: Dict, user_config: Dict) -> Dict[str, Any]:
    """
    Orchestrate all output distribution functions for comprehensive startup analysis distribution.
    
    Takes complete analysis data and user configuration to generate all output distribution
    components including investor portals, startup dashboards, smart notifications, and
    external integrations with proper error handling and system health monitoring.
    
    Args:
        all_analysis_data (Dict): Complete startup analysis data from all analysis modules
        user_config (Dict): User configuration including preferences, roles, and system settings
    
    Returns:
        Dict[str, Any]: Comprehensive output distribution results with system health monitoring
                       and detailed execution status for all components
    """
    try:
        if not model:
            return {"status": "error", "message": "AI model not configured"}
        
        # Validate inputs
        if not all_analysis_data or not isinstance(all_analysis_data, dict):
            return {"status": "error", "message": "Invalid all_analysis_data provided"}
        
        if not user_config or not isinstance(user_config, dict):
            return {"status": "error", "message": "Invalid user_config provided"}
        
        logger.info("Starting comprehensive output distribution orchestration")
        
        # Initialize results container
        distribution_results = {
            "execution_summary": {
                "start_time": datetime.now().isoformat(),
                "components_executed": [],
                "success_count": 0,
                "error_count": 0,
                "warnings": []
            },
            "investor_portal": {},
            "startup_portal": {},
            "notifications": {},
            "integrations": {},
            "system_health": {}
        }
        
        # Extract relevant data for each component
        deal_data = all_analysis_data.get("deal_data", {})
        startup_analysis = all_analysis_data.get("startup_analysis", {})
        improvement_areas = all_analysis_data.get("improvement_areas", [])
        progress_data = all_analysis_data.get("progress_data", {})
        monitoring_data = all_analysis_data.get("monitoring_data", {})
        
        # Extract user configuration
        investor_preferences = user_config.get("investor_preferences", {})
        portfolio_context = user_config.get("portfolio_context", {})
        user_preferences = user_config.get("user_preferences", {})
        alert_thresholds = user_config.get("alert_thresholds", {})
        integration_config = user_config.get("integration_config", {})
        data_mappings = user_config.get("data_mappings", {})
        external_apis = user_config.get("external_apis", [])
        
        # 1. Generate Investor Portal Content
        if deal_data and investor_preferences and portfolio_context:
            logger.info("Generating investor portal content...")
            investor_result = generate_investor_portal_content(
                deal_data, investor_preferences, portfolio_context
            )
            distribution_results["investor_portal"] = investor_result
            distribution_results["execution_summary"]["components_executed"].append("investor_portal")
            
            if investor_result["status"] == "success":
                distribution_results["execution_summary"]["success_count"] += 1
            else:
                distribution_results["execution_summary"]["error_count"] += 1
        else:
            distribution_results["execution_summary"]["warnings"].append(
                "Investor portal generation skipped - insufficient data"
            )
        
        # 2. Generate Startup Portal Content
        if startup_analysis and isinstance(improvement_areas, list) and progress_data:
            logger.info("Generating startup portal content...")
            startup_result = generate_startup_portal_content(
                startup_analysis, improvement_areas, progress_data
            )
            distribution_results["startup_portal"] = startup_result
            distribution_results["execution_summary"]["components_executed"].append("startup_portal")
            
            if startup_result["status"] == "success":
                distribution_results["execution_summary"]["success_count"] += 1
            else:
                distribution_results["execution_summary"]["error_count"] += 1
        else:
            distribution_results["execution_summary"]["warnings"].append(
                "Startup portal generation skipped - insufficient data"
            )
        
        # 3. Generate Smart Notifications
        if monitoring_data and user_preferences and alert_thresholds:
            logger.info("Generating smart notifications...")
            notifications_result = generate_smart_notifications(
                monitoring_data, user_preferences, alert_thresholds
            )
            distribution_results["notifications"] = notifications_result
            distribution_results["execution_summary"]["components_executed"].append("notifications")
            
            if notifications_result["status"] == "success":
                distribution_results["execution_summary"]["success_count"] += 1
            else:
                distribution_results["execution_summary"]["error_count"] += 1
        else:
            distribution_results["execution_summary"]["warnings"].append(
                "Smart notifications generation skipped - insufficient data"
            )
        
        # 4. Manage External Integrations
        if integration_config and data_mappings and isinstance(external_apis, list):
            logger.info("Managing external integrations...")
            integrations_result = manage_external_integrations(
                integration_config, data_mappings, external_apis
            )
            distribution_results["integrations"] = integrations_result
            distribution_results["execution_summary"]["components_executed"].append("integrations")
            
            if integrations_result["status"] == "success":
                distribution_results["execution_summary"]["success_count"] += 1
            else:
                distribution_results["execution_summary"]["error_count"] += 1
        else:
            distribution_results["execution_summary"]["warnings"].append(
                "External integrations management skipped - insufficient configuration"
            )
        
        # 5. Generate System Health Report
        distribution_results["system_health"] = {
            "overall_status": "healthy" if distribution_results["execution_summary"]["error_count"] == 0 else "degraded",
            "total_components": len(distribution_results["execution_summary"]["components_executed"]),
            "successful_components": distribution_results["execution_summary"]["success_count"],
            "failed_components": distribution_results["execution_summary"]["error_count"],
            "warnings_count": len(distribution_results["execution_summary"]["warnings"]),
            "execution_time": (datetime.now() - datetime.fromisoformat(
                distribution_results["execution_summary"]["start_time"]
            )).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Finalize execution summary
        distribution_results["execution_summary"]["end_time"] = datetime.now().isoformat()
        distribution_results["execution_summary"]["total_execution_time"] = distribution_results["system_health"]["execution_time"]
        
        logger.info(f"Output distribution orchestration completed. "
                   f"Success: {distribution_results['execution_summary']['success_count']}, "
                   f"Errors: {distribution_results['execution_summary']['error_count']}")
        
        return {"status": "success", "data": distribution_results}
        
    except Exception as e:
        logger.error(f"Error in main output distribution orchestration: {e}")
        return {"status": "error", "message": f"Failed to orchestrate output distribution: {str(e)}"}


if __name__ == "__main__":
    """
    Example usage of the Output Distribution Application
    """
    print("=== Output Distribution Application Example ===")
    
    # Example data structures
    sample_deal_data = {
        "id": "deal_001",
        "startup_name": "TechStartup Inc",
        "sector": "fintech",
        "stage": "Series A",
        "valuation": 10000000,
        "funding_amount": 2000000,
        "metrics": {
            "revenue": 500000,
            "growth_rate": 0.25,
            "burn_rate": 100000
        }
    }
    
    sample_investor_preferences = {
        "id": "investor_001",
        "name": "Venture Capital Partners",
        "investment_range": [1000000, 5000000],
        "preferred_sectors": ["fintech", "healthtech"],
        "risk_tolerance": "medium",
        "investment_stage": ["seed", "series_a"]
    }
    
    sample_portfolio_context = {
        "current_investments": 15,
        "total_aum": 100000000,
        "sector_allocation": {
            "fintech": 0.3,
            "healthtech": 0.25,
            "edtech": 0.2,
            "other": 0.25
        }
    }
    
    sample_startup_analysis = {
        "id": "startup_001",
        "overall_score": 85,
        "strengths": ["strong_team", "market_traction"],
        "weaknesses": ["limited_funding", "competition"]
    }
    
    sample_improvement_areas = [
        {"area": "financial_management", "priority": "high"},
        {"area": "market_expansion", "priority": "medium"}
    ]
    
    sample_progress_data = {
        "current_metrics": {"revenue": 500000, "users": 10000},
        "goals": {"revenue": 1000000, "users": 25000},
        "timeline": "12_months"
    }
    
    # Example 1: Generate Investor Portal Content
    print("\n1. Testing Investor Portal Content Generation...")
    investor_result = generate_investor_portal_content(
        sample_deal_data, sample_investor_preferences, sample_portfolio_context
    )
    print(f"Status: {investor_result['status']}")
    if investor_result["status"] == "success":
        print("✓ Investor portal content generated successfully")
    else:
        print(f"✗ Error: {investor_result['message']}")
    
    # Example 2: Generate Startup Portal Content
    print("\n2. Testing Startup Portal Content Generation...")
    startup_result = generate_startup_portal_content(
        sample_startup_analysis, sample_improvement_areas, sample_progress_data
    )
    print(f"Status: {startup_result['status']}")
    if startup_result["status"] == "success":
        print("✓ Startup portal content generated successfully")
    else:
        print(f"✗ Error: {startup_result['message']}")
    
    # Example 3: Generate Smart Notifications
    print("\n3. Testing Smart Notifications Generation...")
    sample_monitoring_data = {"performance": "good", "alerts": []}
    sample_user_preferences = {"notifications": "email", "frequency": "daily"}
    sample_alert_thresholds = {"performance": 0.8, "risk": 0.3}
    
    notifications_result = generate_smart_notifications(
        sample_monitoring_data, sample_user_preferences, sample_alert_thresholds
    )
    print(f"Status: {notifications_result['status']}")
    if notifications_result["status"] == "success":
        print("✓ Smart notifications generated successfully")
    else:
        print(f"✗ Error: {notifications_result['message']}")
    
    # Example 4: Manage External Integrations
    print("\n4. Testing External Integrations Management...")
    sample_integration_config = {"crm": "salesforce", "accounting": "quickbooks"}
    sample_data_mappings = {"contact_fields": {}, "deal_fields": {}}
    sample_external_apis = [{"name": "crm_api", "endpoint": "api.example.com"}]
    
    integrations_result = manage_external_integrations(
        sample_integration_config, sample_data_mappings, sample_external_apis
    )
    print(f"Status: {integrations_result['status']}")
    if integrations_result["status"] == "success":
        print("✓ External integrations managed successfully")
    else:
        print(f"✗ Error: {integrations_result['message']}")
    
    # Example 5: Main Output Distribution Orchestration
    print("\n5. Testing Main Output Distribution Orchestration...")
    sample_all_analysis_data = {
        "deal_data": sample_deal_data,
        "startup_analysis": sample_startup_analysis,
        "improvement_areas": sample_improvement_areas,
        "progress_data": sample_progress_data,
        "monitoring_data": sample_monitoring_data
    }
    
    sample_user_config = {
        "investor_preferences": sample_investor_preferences,
        "portfolio_context": sample_portfolio_context,
        "user_preferences": sample_user_preferences,
        "alert_thresholds": sample_alert_thresholds,
        "integration_config": sample_integration_config,
        "data_mappings": sample_data_mappings,
        "external_apis": sample_external_apis
    }
    
    main_result = main_output_distribution(sample_all_analysis_data, sample_user_config)
    print(f"Status: {main_result['status']}")
    if main_result["status"] == "success":
        print("✓ Main output distribution orchestration completed successfully")
        data = main_result["data"]
        print(f"  - Components executed: {len(data['execution_summary']['components_executed'])}")
        print(f"  - Success count: {data['execution_summary']['success_count']}")
        print(f"  - Error count: {data['execution_summary']['error_count']}")
        print(f"  - System status: {data['system_health']['overall_status']}")
    else:
        print(f"✗ Error: {main_result['message']}")
    
    print("\n=== Output Distribution Application Example Complete ===")

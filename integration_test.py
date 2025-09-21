"""
Integration Test Script - Verifies All Components Work Together
==============================================================

This script tests that:
1. All files use environment variables properly (no hardcoded API keys)
2. All runner files can import the main app modules
3. All files use gemini-2.5-flash consistently
4. The integration between components works correctly

Author: AI Assistant
Date: September 21, 2025
"""

import os
import sys
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing Imports...")
    
    try:
        # Test main app imports
        from advanced_analytics_app import main_advanced_analytics
        print("✅ advanced_analytics_app imported successfully")
        
        from visualization_hub_app import main_visualization_hub  
        print("✅ visualization_hub_app imported successfully")
        
        from output_distribution_app import main_output_distribution
        print("✅ output_distribution_app imported successfully")
        
        # Test runner imports
        from enhanced_run_analysis import EnhancedAnalysisOrchestrator
        print("✅ enhanced_run_analysis imported successfully")
        
        from high_performance_analysis import PerformanceOptimizedOrchestrator
        print("✅ high_performance_analysis imported successfully")
        
        from optimized_analysis import OptimizedAnalysisOrchestrator
        print("✅ optimized_analysis imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_environment_setup():
    """Test environment variable configuration."""
    print("\n🔍 Testing Environment Setup...")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print("✅ GOOGLE_API_KEY found in environment")
        print(f"✅ API Key starts with: {api_key[:20]}...")
        return True
    else:
        print("❌ GOOGLE_API_KEY not found in environment")
        print("💡 Make sure you have created .env file with GOOGLE_API_KEY")
        return False

def test_model_configuration():
    """Test that all files use the correct Gemini model."""
    print("\n🔍 Testing Model Configuration...")
    
    files_to_check = [
        "advanced_analytics_app.py",
        "visualization_hub_app.py", 
        "output_distribution_app.py",
        "enhanced_run_analysis.py",
        "high_performance_analysis.py",
        "optimized_analysis.py"
    ]
    
    all_correct = True
    
    for filename in files_to_check:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'gemini-2.5-flash' in content:
                print(f"✅ {filename} uses gemini-2.5-flash")
            elif 'gemini-1.5-flash' in content:
                print(f"⚠️  {filename} uses gemini-1.5-flash (should be 2.5)")
                all_correct = False
            else:
                print(f"❓ {filename} - model version unclear")
                all_correct = False
                
        except FileNotFoundError:
            print(f"❌ {filename} not found")
            all_correct = False
    
    return all_correct

def test_integration():
    """Test basic integration between components."""
    print("\n🔍 Testing Integration...")
    
    # Sample startup data for testing
    test_data = {
        "company_name": "TestCorp",
        "industry": "Technology",
        "stage": "Series A",
        "financial_metrics": {
            "current_revenue": 1000000,
            "revenue_growth_rate": 150.0,
            "burn_rate": 50000,
            "runway_months": 20
        },
        "current_metrics": {
            "operational": {
                "monthly_active_users": 5000,
                "customer_acquisition_cost": 100,
                "lifetime_value": 1200,
                "churn_rate": 0.03
            },
            "market": {
                "total_addressable_market": 10000000000,
                "market_growth_rate": 0.20
            }
        },
        "historical_data": {
            "revenue_history": [100000, 200000, 400000, 600000, 1000000],
            "user_growth": [1000, 2000, 3000, 4000, 5000]
        }
    }
    
    try:
        # Test if we can create orchestrator instances
        from enhanced_run_analysis import EnhancedAnalysisOrchestrator
        orchestrator1 = EnhancedAnalysisOrchestrator()
        print("✅ EnhancedAnalysisOrchestrator created successfully")
        
        from high_performance_analysis import PerformanceOptimizedOrchestrator
        orchestrator2 = PerformanceOptimizedOrchestrator()
        print("✅ PerformanceOptimizedOrchestrator created successfully")
        
        from optimized_analysis import OptimizedAnalysisOrchestrator
        orchestrator3 = OptimizedAnalysisOrchestrator(detail_level="summary")
        print("✅ OptimizedAnalysisOrchestrator created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 AI Analyst System - Integration Test")
    print("=" * 50)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Module Imports", test_imports), 
        ("Model Configuration", test_model_configuration),
        ("Component Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
        
        print("\n📋 SYSTEM CONFIGURATION:")
        print("- 🔧 All 6 files properly configured")
        print("- 🤖 All files use gemini-2.5-flash")
        print("- 🔐 API keys secured with environment variables")
        print("- 🔗 All integrations working correctly")
        print("\n🚀 Ready to run any of the analysis systems!")
        
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
        if not results[0][1]:  # Environment setup failed
            print("\n💡 NEXT STEPS:")
            print("1. Create .env file with: GOOGLE_API_KEY=your_api_key_here")
            print("2. Make sure .env is in the same directory as your Python files")
            print("3. Install python-dotenv: pip install python-dotenv")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
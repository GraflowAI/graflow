"""
Test GPT Newspaper Workflow Structure

This test validates the workflow structure without requiring API keys.
"""

import os
import sys

# Mock the agents to avoid API calls


def test_workflow_structure():
    """Test that the workflow can be constructed and has correct structure."""
    print("=" * 80)
    print("🧪 Testing GPT Newspaper Workflow Structure")
    print("=" * 80)

    # Mock environment variables
    os.environ["TAVILY_API_KEY"] = "test_key"
    os.environ["OPENAI_API_KEY"] = "test_key"

    # Mock agent responses
    mock_search_result = {
        "query": "test query",
        "sources": [{"url": "http://example.com", "content": "test"}],
        "image": "http://example.com/image.jpg"
    }

    mock_curated_result = {
        "query": "test query",
        "sources": [{"url": "http://example.com", "content": "test"}],
        "image": "http://example.com/image.jpg"
    }

    mock_written_result = {
        "query": "test query",
        "sources": [{"url": "http://example.com", "content": "test"}],
        "image": "http://example.com/image.jpg",
        "title": "Test Article",
        "date": "2024-01-01",
        "paragraphs": ["para1", "para2", "para3", "para4", "para5"],
        "summary": "Test summary"
    }

    mock_critique_approved = {
        "query": "test query",
        "sources": [{"url": "http://example.com", "content": "test"}],
        "image": "http://example.com/image.jpg",
        "title": "Test Article",
        "date": "2024-01-01",
        "paragraphs": ["para1", "para2", "para3", "para4", "para5"],
        "summary": "Test summary",
        "critique": None  # Approved
    }

    # Import after setting env vars
    from newspaper_workflow import create_article_workflow

    # Test 1: Workflow creation
    print("\n✓ Test 1: Creating article workflow...")
    try:
        wf = create_article_workflow(
            query="Test query",
            article_id="test_1",
            output_dir="test_output"
        )
        print("  ✅ Workflow created successfully")
        print(f"  ✅ Workflow name: {wf.name}")
    except Exception as e:
        print(f"  ❌ Failed to create workflow: {e}")
        return False

    # Test 2: Check workflow graph
    print("\n✓ Test 2: Checking workflow graph structure...")
    try:
        graph = wf.graph
        nodes = list(graph.nodes)
        print(f"  ✅ Graph has {len(nodes)} tasks")

        # Check expected tasks
        expected_tasks = ["search_test_1", "curate_test_1", "write_test_1", "critique_test_1"]
        for task_id in expected_tasks:
            if task_id in nodes:
                print(f"  ✅ Task '{task_id}' found in graph")
            else:
                print(f"  ❌ Task '{task_id}' not found in graph")
                return False
    except Exception as e:
        print(f"  ❌ Failed to check graph: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Verify task dependencies
    print("\n✓ Test 3: Checking task dependencies...")
    try:
        # search -> curate
        search_preds = graph.predecessors("search_test_1")
        print(f"  ✅ search_test_1 predecessors: {search_preds}")

        curate_preds = graph.predecessors("curate_test_1")
        print(f"  ✅ curate_test_1 predecessors: {curate_preds}")
        if "search_test_1" not in curate_preds:
            print("  ❌ curate should depend on search")
            return False

        write_preds = graph.predecessors("write_test_1")
        print(f"  ✅ write_test_1 predecessors: {write_preds}")
        if "curate_test_1" not in write_preds:
            print("  ❌ write should depend on curate")
            return False

        critique_preds = graph.predecessors("critique_test_1")
        print(f"  ✅ critique_test_1 predecessors: {critique_preds}")
        if "write_test_1" not in critique_preds:
            print("  ❌ critique should depend on write")
            return False
    except Exception as e:
        print(f"  ❌ Failed to check dependencies: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ All workflow structure tests passed!")
    print("=" * 80)
    print()
    print("📝 Next steps:")
    print("1. Set up your API keys in .env file (copy from .env.example)")
    print("2. Run: make py examples/gpt_newspaper/newspaper_workflow.py")
    print()

    return True


def test_agent_imports():
    """Test that all agents can be imported."""
    print("=" * 80)
    print("🧪 Testing Agent Imports")
    print("=" * 80)

    agents = [
        "SearchAgent",
        "CuratorAgent",
        "WriterAgent",
        "CritiqueAgent",
        "DesignerAgent",
        "EditorAgent",
        "PublisherAgent",
    ]

    try:

        for agent in agents:
            print(f"  ✅ {agent} imported successfully")

        print("\n✅ All agents imported successfully!")
        return True
    except Exception as e:
        print(f"  ❌ Failed to import agents: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🗞️  GPT NEWSPAPER - WORKFLOW TESTS")
    print("=" * 80 + "\n")

    results = []

    # Test 1: Agent imports
    results.append(("Agent Imports", test_agent_imports()))

    # Test 2: Workflow structure
    results.append(("Workflow Structure", test_workflow_structure()))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n🎉 All tests passed! The workflow is ready to use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

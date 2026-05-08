"""
RAG retrieval test script.
Tests the vector store retrieval directly, bypassing the voice pipeline.

Run: python test_rag.py
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env.local"))

from rag import retrieve, get_collection


async def run_tests():
    collection = get_collection()
    print(f"Collection has {collection.count()} chunks\n")

    # Each test has a query and what the correct answer should contain
    tests = [
        {
            "query": "How to make puttu",
            "expected_keywords": ["Kerala", "steamed", "coconut", "rice", "puttu flour", "cylindrical"],
            "note": "Page 51: Puttu is steamed rice cakes layered with coconut from Kerala, needs a cylindrical puttu steamer",
        },
        {
            "query": "neer dosa recipe",
            "expected_keywords": ["water dosa", "Mangalore", "rice", "coconut", "crêpes"],
            "note": "Page 47: Neer dosa means water dosa, gluten-free rice crêpes from Mangalore/Udupi",
        },
        {
            "query": "gongura pachadi",
            "expected_keywords": ["sorrel leaves", "Telugu", "sesame oil", "fenugreek", "garlic"],
            "note": "Page 101: Sorrel leaves chutney, signature Telugu dish, spiced with red chiles and garlic",
        },
        {
            "query": "fish molee Kerala",
            "expected_keywords": ["coconut milk", "Kerala", "Portuguese", "tilapia", "cardamom"],
            "note": "Page 162: Kerala-style fish stew with coconut milk, Portuguese-influenced",
        },
        {
            "query": "pesarattu green mung bean",
            "expected_keywords": ["mung bean", "Andhra", "Telangana", "ginger chutney", "protein"],
            "note": "Page 49: Green mung bean crepes from Andhra/Telangana, served with spicy ginger chutney",
        },
        {
            "query": "vangi bath brinjal rice",
            "expected_keywords": ["eggplant", "spice powder", "cinnamon", "coriander"],
            "note": "Page 72: Brinjal/eggplant rice, best with freshly ground spices, vangi bath spice powder",
        },
        {
            "query": "ven pongal recipe",
            "expected_keywords": ["rice", "mung beans", "peppercorns", "ghee", "cashews", "breakfast"],
            "note": "Page 81: Savory rice with split yellow mung beans, paired with vadai, sambar, chutney",
        },
        {
            "query": "curry leaf mint cilantro chutney",
            "expected_keywords": ["curry leaves", "mint", "cilantro", "tamarind", "idli", "dosai"],
            "note": "Page 97: Chutney with mint and cilantro, perfect for idli and dosai",
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(tests, 1):
        print(f"{'='*60}")
        print(f"TEST {i}: {test['query']}")
        print(f"Expected: {test['note']}")
        print(f"{'─'*60}")

        result = await retrieve(test["query"], n_results=3)

        # Check how many expected keywords appear in the result
        found = [kw for kw in test["expected_keywords"] if kw.lower() in result.lower()]
        missing = [kw for kw in test["expected_keywords"] if kw.lower() not in result.lower()]

        score = len(found) / len(test["expected_keywords"])

        if score >= 0.5:
            print(f"✅ PASS ({score:.0%}) — Found: {found}")
            passed += 1
        else:
            print(f"❌ FAIL ({score:.0%}) — Found: {found}, Missing: {missing}")
            failed += 1

        # Show first 300 chars of retrieved content
        print(f"\nRetrieved (first 300 chars):")
        print(f"{result[:300]}...")
        print()

    print(f"{'='*60}")
    print(f"RESULTS: {passed}/{passed+failed} tests passed")
    if failed > 0:
        print(f"\n⚠️  {failed} tests failed — retrieval quality may need tuning")
        print("Try: increase chunk size, reduce overlap, or increase n_results")
    else:
        print("\n🎉 All tests passed — RAG retrieval is working correctly")


if __name__ == "__main__":
    asyncio.run(run_tests())

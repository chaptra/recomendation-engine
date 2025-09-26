import requests
import json

def test_clean_api():
    base_url = "http://localhost:9000"
    
    print("🎯 TESTING CLEAN BOOK RECOMMENDATION API")
    print("="*50)
    
    # Test 1: Health Check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ API is healthy")
            print(f"   📚 Total books: {data['total_books']}")
            print(f"   🧠 Algorithm: {data['algorithm']}")
            print(f"   🏃 Training required: {data['training_required']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return
    
    # Test 2: Book Recommendations
    print("\n2. Book Recommendations:")
    test_books = ["Moby Dick", "Harry Potter", "1984", "Pride and Prejudice"]
    
    for book_title in test_books:
        print(f"\n   🔍 Testing: '{book_title}'")
        try:
            response = requests.get(
                f"{base_url}/suggest", 
                params={"title": book_title, "limit": 3},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data['found']:
                    print(f"      ✅ Found source book: {data['source_book']['title']}")
                    print(f"      📊 Got {data['total']} recommendations")
                    
                    for i, rec in enumerate(data['suggestions'][:2], 1):
                        print(f"      {i}. {rec['title']}")
                        print(f"         Similarity: {rec['similarity_score']}")
                        print(f"         Reason: {rec['reason']}")
                else:
                    print(f"      ⚠️ No matches found: {data.get('message', 'Unknown')}")
            else:
                print(f"      ❌ Request failed: {response.status_code}")
                print(f"      📝 Error: {response.text[:100]}...")
                
        except requests.exceptions.Timeout:
            print(f"      ⏰ Request timed out (>30s)")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Test 3: API Documentation
    print("\n3. API Documentation:")
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("   ✅ API documentation accessible")
            print(f"   🌐 Docs URL: {base_url}/docs")
        else:
            print(f"   ❌ Docs not accessible: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Docs error: {e}")
    
    print("\n" + "="*50)
    print("🎉 TESTING COMPLETE!")
    print("📍 Key Points:")
    print("   • No ML model training required")
    print("   • Uses content-based filtering")
    print("   • Real-time similarity computation")
    print("   • Works with book titles only")
    print(f"   • API Docs: {base_url}/docs")

if __name__ == "__main__":
    test_clean_api()
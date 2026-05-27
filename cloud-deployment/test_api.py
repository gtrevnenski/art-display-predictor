"""
Test script for the MET Art Display Predictor API
Run this to verify your API is working correctly
"""

import requests
import json

# Update this with your actual API URL after deployment
API_URL = "http://localhost:8000"  # For local testing (docker run -p 8000:8080)
# API_URL = "https://met-art-predictor-xxx-uc.a.run.app"  # For production

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction():
    """Test prediction endpoint"""
    print("Testing /predict endpoint...")
    
    test_cases = [
        {
            "name": "European Painting",
            "data": {
                "text": "Oil painting depicting a pastoral landscape with figures from French countryside",
                "objectEndDate": 1860,
                "department": "European Paintings",
                "country": "France",
                "cat1": "Paintings",
                "subcat1": "Paintings",
                "cat2": "Paintings"
            }
        },
        {
            "name": "Greek Vase",
            "data": {
                "text": "Ancient Greek red-figure vase with mythological scene showing heroes and gods",
                "objectEndDate": -400,
                "department": "Greek and Roman Art",
                "country": "Greece",
                "cat1": "Vases",
                "subcat1": "Pottery",
                "cat2": "None"
            }
        },
        {
            "name": "Modern Sculpture",
            "data": {
                "text": "Abstract bronze sculpture from 20th century American modernist movement",
                "objectEndDate": 1960,
                "department": "Modern and Contemporary Art",
                "country": "United States",
                "cat1": "Sculpture",
                "subcat1": "Sculpture",
                "cat2": "None"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        response = requests.post(
            f"{API_URL}/predict",
            json=test_case['data']
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Prediction: {result['prediction']}")
            print(f"  Probability: {result['probability']:.3f}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Explanation: {result['explanation']}")
        else:
            print(f"[ERROR] Status: {response.status_code}")
            print(f"  {response.text}")
        print()

def test_batch():
    """Test batch prediction endpoint"""
    print("Testing /predict/batch endpoint...")
    
    requests_data = [
        {
            "text": "Oil painting of a landscape with mountains and trees",
            "objectEndDate": 1850,
            "department": "European Paintings",
            "country": "France",
            "cat1": "Paintings",
            "subcat1": "Paintings",
            "cat2": "Paintings"
        },
        {
            "text": "Ancient Egyptian limestone statue of pharaoh",
            "objectEndDate": -1500,
            "department": "Egyptian Art",
            "country": "Egypt",
            "cat1": "Stone Sculpture",
            "subcat1": "Stone",
            "cat2": "None"
        }
    ]
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=requests_data
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {result['count']} predictions")
        for i, pred in enumerate(result['predictions']):
            print(f"  {i+1}. {pred.get('prediction', 'error')}: {pred.get('probability', 'N/A')}")
    print()

def test_performance():
    """Test API response time"""
    import time
    
    print("Testing performance...")
    
    test_data = {
        "text": "Oil painting of a landscape with river and trees",
        "objectEndDate": 1860,
        "department": "European Paintings",
        "country": "France",
        "cat1": "Paintings",
        "subcat1": "Paintings",
        "cat2": "Paintings"
    }
    
    times = []
    num_requests = 5
    
    for i in range(num_requests):
        start = time.time()
        response = requests.post(f"{API_URL}/predict", json=test_data)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Request {i+1}: {elapsed*1000:.0f}ms")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage response time: {avg_time*1000:.0f}ms")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("MET Art Display Predictor - API Tests")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_prediction()
        test_batch()
        test_performance()
        
        print("=" * 60)
        print("[SUCCESS] All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to API")
        print(f"Make sure the API is running at: {API_URL}")
        print("\nFor local testing:")
        print("  python app.py")
        print("\nOr with Docker:")
        print("  docker run -p 8000:8080 met-predictor")
    except Exception as e:
        print(f"[ERROR] {e}")


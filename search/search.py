import requests
import os

def search_google(query, num_results=5):
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("SEARCH_ENGINE_ID")
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={search_engine_id}&q={query}&num={num_results}"


    try:
        response = requests.get(url)
        response.raise_for_status()
        search_results = response.json()

        results = []
        for item in search_results.get('items', []):
            results.append({
                'Title': item.get('title', ''),
                'Text': item.get('snippet', '')
            })

        return results

    except requests.RequestException as e:
        print(f"Error fetching search results: {e}")
        return []

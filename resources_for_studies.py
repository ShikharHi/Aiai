# Import the required libraries
from uyts import Search
import requests

def youtube_search(query, min_results=5, language='en', country='GB'):
    # Create a Search object with the specified parameters
    search = Search(query, minResults=min_results, language=language, country=country)

    # List to store video links
    video_links = []

    # Print the search results
    print(f"Search results for '{query}':\n")
    for result in search.results:
        if result.resultType == 'video':
            # Construct the YouTube video URL
            video_url = f"https://www.youtube.com/watch?v={result.id}"
            video_links.append(video_url)
            # Stop after 5 video links
            if len(video_links) >= 5:
                break

    return video_links

def fetch_wikipedia_links(topic):
    """
    Fetch relevant Wikipedia links based on the given topic.
    
    Parameters:
    topic (str): The topic of study.

    Returns:
    list: A list of suggested links from Wikipedia.
    """
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": 5,  # Limit the number of results
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    links = []
    for item in data.get("query", {}).get("search", []):
        title = item["title"]
        link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        links.append(link)

    return links if links else ["Sorry, I couldn't find any resources for that topic."]

# Main execution
if __name__ == "__main__":
    topic = input("Enter the topic you are studying: ")
    
    # Fetch and display Wikipedia links
    suggestions = fetch_wikipedia_links(topic)
    print("\nSuggested Resources:")
    for resource in suggestions:
        print(f"- {resource}")

    # Perform YouTube search
    links = youtube_search(topic, min_results=10, language='en', country='US')
    print("\nLinks to the top 5 YouTube videos:")
    for link in links:
        print(link)  # This will print the YouTube video links
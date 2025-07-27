import requests
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                          (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        }

    def scrape_paragraphs(self, url):
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            return [p.get_text(strip=True) for p in paragraphs]
        else:
            print(f"Request failed with status code: {response.status_code}")
            return []

# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    url = "https://mgmt.iisc.ac.in/list-of-courses/"
    paragraphs = scraper.scrape_paragraphs(url)
    
    for i, text in enumerate(paragraphs, 1):
        print(f"{i}. {text}\n")
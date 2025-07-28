import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup


class WebScraper:
    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/115.0 Safari/537.36"
            )
        }

    def scrape_paragraphs(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
        except RequestException as e:
            print(f"Request error when trying to access {url}: {e}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        texts = [p.get_text(strip=True)
                 for p in paragraphs if p.get_text(strip=True)]
        return texts


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    url = "https://mgmt.iisc.ac.in/master-management/"
    paragraphs = scraper.scrape_paragraphs(url)

    for i, text in enumerate(paragraphs, 1):
        print(f"{i}. {text}\n")

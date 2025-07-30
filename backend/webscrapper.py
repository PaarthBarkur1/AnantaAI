import requests
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PageElement
import logging
from urllib.parse import urljoin
from typing import Optional, List, Dict, Union, Any, cast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebScraper:
    def __init__(self, user_agent=None, default_timeout=10):
        self.headers = {
            "User-Agent": user_agent or (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/115.0 Safari/537.36"
            )
        }
        self.default_timeout = default_timeout

    def _make_request(self, url: str) -> requests.Response | None:
        """Helper method to make an HTTP GET request and handle common errors."""
        try:
            response = requests.get(url, headers=self.headers, timeout=self.default_timeout)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response
        except Timeout:
            logging.warning(f"Timeout occurred while accessing {url}")
            return None
        except HTTPError as e:
            logging.warning(f"HTTP error {e.response.status_code} occurred for {url}: {e}")
            return None
        except ConnectionError as e:
            logging.warning(f"Connection error occurred for {url}: {e}")
            return None
        except RequestException as e:
            logging.error(f"An unexpected request error occurred for {url}: {e}")
            return None

    def scrape_html(self, url: str) -> BeautifulSoup | None:
        """Fetches a URL and returns a BeautifulSoup object."""
        response = self._make_request(url)
        if response:
            return BeautifulSoup(response.content, "html.parser")
        return None

    def scrape_paragraphs(self, url: str) -> list[str]:
        """Scrapes all non-empty paragraph texts from a given URL."""
        soup = self.scrape_html(url)
        if not soup:
            return []

        paragraphs = soup.find_all("p")
        texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        logging.info(f"Scraped {len(texts)} paragraphs from {url}")
        return texts

    def scrape_headlines(self, url: str, levels: tuple[int, ...] = (1, 2, 3)) -> list[dict]:
        """Scrapes headlines (h1, h2, h3, etc.) and their texts."""
        soup = self.scrape_html(url)
        if not soup:
            return []

        headlines = []
        for level in levels:
            for heading in soup.find_all(f"h{level}"):
                text = heading.get_text(strip=True)
                if text:
                    headlines.append({"level": level, "text": text})
        logging.info(f"Scraped {len(headlines)} headlines from {url}")
        return headlines

    def scrape_links(self, url: str, selector: str = "a") -> List[Dict[str, str]]:
        """Scrapes all links (href and text) from a given URL, optionally filtered by a CSS selector."""
        soup = self.scrape_html(url)
        if not soup:
            return []

        links = []
        for a_tag in soup.select(selector):
            if isinstance(a_tag, Tag):  # Type check to ensure we have a Tag
                href = a_tag.get("href")
                text = a_tag.get_text(strip=True)
                if href and text:
                    # Ensure href is str before using urljoin
                    href_str = str(href)
                    absolute_href = urljoin(url, href_str)
                    links.append({"text": text, "href": absolute_href})
        logging.info(f"Scraped {len(links)} links from {url}")
        return links

    def scrape_table_data(self, url: str, table_selector: str = "table") -> List[List[List[str]]]:
        """
        Scrapes data from tables. Returns a list of tables, where each table is
        a list of rows, and each row is a list of cell texts.
        """
        soup = self.scrape_html(url)
        if not soup:
            return []

        all_tables_data = []
        for table in soup.select(table_selector):
            if isinstance(table, Tag):  # Type check to ensure we have a Tag
                table_data = []
                rows = table.find_all("tr")
                for row in rows:
                    if isinstance(row, Tag):  # Type check for row
                        cols = row.find_all(["td", "th"])
                        row_data = [col.get_text(strip=True) for col in cols if isinstance(col, Tag)]
                        if row_data: # Only add non-empty rows
                            table_data.append(row_data)
                if table_data:
                    all_tables_data.append(table_data)
        logging.info(f"Scraped {len(all_tables_data)} tables from {url}")
        return all_tables_data

    def scrape_wordpress_recent_posts(self, url: str, max_posts: int = 5) -> List[Dict[str, str]]:
        """Scrape recent post titles and links from a WordPress site homepage."""
        soup = self.scrape_html(url)
        if not soup:
            return []

        posts = []
        # Try to find posts in common WordPress structures
        # Prioritize more specific article/post containers first
        for entry in soup.select('article.post, div.post, .entry-content a, .post-content a'):
            if isinstance(entry, Tag):  # Type check for entry
                a_tag = entry.find('a', href=True)  # Find an 'a' tag with an href within the entry
                if isinstance(a_tag, Tag):  # Type check for a_tag
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href')
                    if href:  # Check if href exists
                        link = urljoin(url, str(href))
                        if title and link and {"title": title, "link": link} not in posts:  # Avoid duplicates
                            posts.append({"title": title, "link": link})
                            if len(posts) >= max_posts:
                                break

        if len(posts) < max_posts:  # If not enough posts found yet, try other common areas
            # Fallback: try to find links in widgets or lists, often in sidebars or footers
            for a_tag in soup.select('.widget_recent_entries a, .recent-posts a, #recent-posts-2 li a'):
                if isinstance(a_tag, Tag):  # Type check for a_tag
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href')
                    if href:  # Check if href exists
                        link = urljoin(url, str(href))
                        if title and link and {"title": title, "link": link} not in posts:  # Avoid duplicates
                            posts.append({"title": title, "link": link})
                            if len(posts) >= max_posts:
                                break
        logging.info(f"Scraped {len(posts)} recent posts from {url}")
        return posts

    def scrape_specific_element(self, url: str, css_selector: str) -> list[str]:
        """
        Scrapes text content from elements matching a given CSS selector.
        Returns a list of stripped text content.
        """
        soup = self.scrape_html(url)
        if not soup:
            return []

        elements = soup.select(css_selector)
        texts = [element.get_text(strip=True) for element in elements if element.get_text(strip=True)]
        logging.info(f"Scraped {len(texts)} elements with selector '{css_selector}' from {url}")
        return texts

# # Example Usage with your provided URLs
# if __name__ == "__main__":
# scraper = WebScraper()

# # Define the URLs from your list
# iisc_urls = [
#     {"name": "IISc M.Mgt Program Overview", "url": "https://mgmt.iisc.ac.in/programmes/master-of-management/", "category": ["program", "overview", "curriculum", "structure", "courses"]},
#     {"name": "IISc Admissions", "url": "https://admissions.iisc.ac.in/", "category": ["admission", "eligibility", "application", "cutoff", "percentile", "cat", "gate"]},
#     {"name": "IISc Placements", "url": "https://mgmt.iisc.ac.in/placements/", "category": ["placement", "salary", "ctc", "companies", "offers"]},
#     {"name": "IISc Faculty & Research", "url": "https://mgmt.iisc.ac.in/faculty/", "category": ["faculty", "research", "projects", "phd", "supervisor", "ongoing"]},
#     {"name": "IISc Student Life", "url": "https://mgmt.iisc.ac.in/student-life/", "category": ["student life", "campus", "hostel", "accommodation", "facilities"]},
#     {"name": "IISc M.Mgt Main Page", "url": "https://mgmt.iisc.ac.in/mmgt/", "category": ["program", "overview", "m.mgt", "main", "home"]},
#     {"name": "IISc Campus Life", "url": "https://mgmt.iisc.ac.in/campus-life/", "category": ["campus", "student life", "facilities", "environment"]},
#     {"name": "IISc Location", "url": "https://mgmt.iisc.ac.in/locate-us/", "category": ["location", "address", "map", "directions"]},
#     {"name": "IISc About Us", "url": "https://mgmt.iisc.ac.in/about-us/", "category": ["about", "institute", "history", "mission"]},
#     {"name": "IISc Faculty Directory", "url": "https://mgmt.iisc.ac.in/faculty-2/", "category": ["faculty", "directory", "professors", "staff"]},
#     {"name": "IISc Management News & Updates", "url": "https://mgmt.iisc.ac.in/newwordpress/", "category": ["news", "updates", "announcements", "recent posts", "events", "blog"]}
# ]

# print("\n--- Scraping Paragraphs ---")
# for item in iisc_urls:
#     if "program" in item["category"] or "about" in item["category"] or "student life" in item["category"]:
#         print(f"\nScraping paragraphs from: {item['url']}")
#         paragraphs = scraper.scrape_paragraphs(item['url'])
#         # for p in paragraphs[:3]: # Print first 3 paragraphs for brevity
#         #     print(p)
#         # if paragraphs:
#         #     print(f"...and {len(paragraphs) - 3} more paragraphs.")
#         # else:
#         #     print("No paragraphs found.")


# print("\n--- Scraping Headlines ---")
# for item in iisc_urls:
#     if "program" in item["category"] or "faculty" in item["category"]:
#         print(f"\nScraping headlines from: {item['url']}")
#         headlines = scraper.scrape_headlines(item['url'])
#         # for h in headlines[:3]: # Print first 3 headlines for brevity
#         #     print(h)
#         # if headlines:
#         #     print(f"...and {len(headlines) - 3} more headlines.")
#         # else:
#         #     print("No headlines found.")

# print("\n--- Scraping Links (e.g., Admissions page) ---")
# admissions_url = next((item['url'] for item in iisc_urls if item['name'] == "IISc Admissions"), None)
# if admissions_url:
#     print(f"\nScraping links from: {admissions_url}")
#     links = scraper.scrape_links(admissions_url)
#     # for link in links[:5]:
#     #     print(link)
#     # if links:
#     #     print(f"...and {len(links) - 5} more links.")
#     # else:
#     #     print("No links found.")

# print("\n--- Scraping WordPress Recent Posts ---")
# news_url = next((item['url'] for item in iisc_urls if "news" in item["category"]), None)
# if news_url:
#     print(f"\nScraping recent posts from: {news_url}")
#     recent_posts = scraper.scrape_wordpress_recent_posts(news_url, max_posts=3)
#     # for post in recent_posts:
#     #     print(f"Title: {post['title']}, Link: {post['link']}")
#     # if not recent_posts:
#     #     print("No recent posts found.")

# print("\n--- Scraping Table Data (e.g., Placements page - might need specific selectors) ---")
# placements_url = next((item['url'] for item in iisc_urls if item['name'] == "IISc Placements"), None)
# if placements_url:
#     print(f"\nScraping tables from: {placements_url}")
#     tables_data = scraper.scrape_table_data(placements_url)
#     # if tables_data:
#     #     for i, table in enumerate(tables_data):
#     #         print(f"Table {i+1} (first 3 rows):")
#     #         for row in table[:3]:
#     #             print(row)
#     #         if len(table) > 3:
#     #             print(f"...and {len(table) - 3} more rows.")
#     # else:
#     #     print("No tables found.")

# print("\n--- Scraping Specific Element (e.g., a specific div or class) ---")
# # This is an example, you would need to inspect the actual webpage to find a useful selector
# program_overview_url = next((item['url'] for item in iisc_urls if item['name'] == "IISc M.Mgt Program Overview"), None)
# if program_overview_url:
#     print(f"\nScraping specific elements from: {program_overview_url} (e.g., elements with class 'content-area')")
#     # You'll need to inspect the page's HTML to find relevant classes/IDs
#     # For demonstration, let's assume there's a div with class 'main-content'
#     specific_content = scraper.scrape_specific_element(program_overview_url, "div.entry-content")
#     # for text in specific_content[:2]:
#     #     print(text)
#     # if not specific_content:
#     #     print("No elements found with the given selector.")
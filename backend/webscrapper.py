import requests
from requests.exceptions import RequestException, HTTPError, Timeout, ConnectionError
from bs4 import BeautifulSoup, Tag
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
            if isinstance(a_tag, Tag):
                href = a_tag.get("href")
                text = a_tag.get_text(strip=True)
                if href and text:
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
            if isinstance(table, Tag):
                table_data = []
                rows = table.find_all("tr")
                for row in rows:
                    if isinstance(row, Tag):
                        cols = row.find_all(["td", "th"])
                        row_data = [col.get_text(strip=True) for col in cols if isinstance(col, Tag)]
                        if row_data:
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
        for entry in soup.select('article.post, div.post, .entry-content a, .post-content a'):
            if isinstance(entry, Tag):
                a_tag = entry.find('a', href=True)
                if isinstance(a_tag, Tag):
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href')
                    if href:
                        link = urljoin(url, str(href))
                        if title and link and {"title": title, "link": link} not in posts:
                            posts.append({"title": title, "link": link})
                            if len(posts) >= max_posts:
                                break

        if len(posts) < max_posts:
            for a_tag in soup.select('.widget_recent_entries a, .recent-posts a, #recent-posts-2 li a'):
                if isinstance(a_tag, Tag):
                    title = a_tag.get_text(strip=True)
                    href = a_tag.get('href')
                    if href:
                        link = urljoin(url, str(href))
                        if title and link and {"title": title, "link": link} not in posts:
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
    
    # --- ENHANCED METHODS BASED ON YOUR FILES ---

    def scrape_landing_page_content(self, url: str) -> Dict[str, Any]:
        """Scrapes specific content from the landing_page.html structure."""
        soup = self.scrape_html(url)
        if not soup:
            return {}
        
        data = {}
        
        # Hero section: Title and Subtitle
        hero_section = soup.find('section', class_='hero-section')
        if hero_section:
            data['hero_title'] = hero_section.find('h1').get_text(strip=True) if hero_section.find('h1') else None
            data['hero_subtitle'] = hero_section.find('p', class_='text-xl').get_text(strip=True) if hero_section.find('p', class_='text-xl') else None
        
        # Key statistics
        stats_section = soup.find('section', id='stats')
        if stats_section:
            data['statistics'] = []
            for stat in stats_section.find_all('div', class_='text-center'):
                value = stat.find('div', class_='text-4xl').get_text(strip=True) if stat.find('div', class_='text-4xl') else None
                label = stat.find('p').get_text(strip=True) if stat.find('p') else None
                if value and label:
                    data['statistics'].append({'value': value, 'label': label})
                    
        # Testimonials
        testimonials_section = soup.find('section', id='testimonials')
        if testimonials_section:
            data['testimonials'] = []
            for testimonial in testimonials_section.find_all('div', class_='bg-white/5'):
                quote = testimonial.find('p', class_='italic').get_text(strip=True) if testimonial.find('p', class_='italic') else None
                author = testimonial.find('p', class_='font-bold').get_text(strip=True) if testimonial.find('p', class_='font-bold') else None
                if quote and author:
                    data['testimonials'].append({'quote': quote, 'author': author})
        
        logging.info(f"Scraped custom landing page content from {url}")
        return data

    def scrape_recruiter_insights_page(self, url: str) -> Dict[str, Any]:
        """Scrapes specific content from the recruiter-insights.html structure."""
        soup = self.scrape_html(url)
        if not soup:
            return {}
        
        data = {}
        
        data['page_title'] = soup.find('h1').get_text(strip=True) if soup.find('h1') else None
        
        insights_container = soup.find('div', class_='space-y-12')
        if insights_container:
            data['insights'] = []
            for insight in insights_container.find_all('div', class_='p-6'):
                title = insight.find('h3').get_text(strip=True) if insight.find('h3') else None
                description = insight.find('p').get_text(strip=True) if insight.find('p') else None
                if title and description:
                    data['insights'].append({'title': title, 'description': description})
        
        logging.info(f"Scraped custom recruiter insights page content from {url}")
        return data

    def scrape_contact_page_content(self, url: str) -> Dict[str, Any]:
        """Scrapes specific content from the contact.html structure."""
        soup = self.scrape_html(url)
        if not soup:
            return {}
            
        data = {}
        
        data['heading'] = soup.find('h1').get_text(strip=True) if soup.find('h1') else None
        
        form = soup.find('form')
        if form:
            data['form_fields'] = []
            for input_field in form.find_all(['input', 'textarea']):
                field_name = input_field.get('name') or input_field.get('id') or input_field.get('placeholder')
                if field_name:
                    data['form_fields'].append(field_name)
        
        contact_info_div = soup.find('div', class_='mt-8')
        if contact_info_div:
            data['contact_info'] = {}
            info_items = contact_info_div.find_all('p', class_='flex')
            for item in info_items:
                icon_name = item.find('svg').get('aria-label') if item.find('svg') else 'info'
                text = item.get_text(strip=True)
                data['contact_info'][icon_name] = text
        
        logging.info(f"Scraped custom contact page content from {url}")
        return data

# Example Usage
if __name__ == "__main__":
    scraper = WebScraper()
    
    # Hypothetical URLs based on your file structure
    base_url = "https://example-doms-placement.com" 
    
    # --- Using the enhanced custom methods ---
    print("\n--- Scraping Custom Landing Page Content ---")
    landing_data = scraper.scrape_landing_page_content(f"{base_url}/pages/landing_page.html")
    print(f"Landing Page Data: {landing_data}")
    
    print("\n--- Scraping Custom Recruiter Insights Page Content ---")
    recruiter_data = scraper.scrape_recruiter_insights_page(f"{base_url}/pages/recruiter-insights.html")
    print(f"Recruiter Insights Data: {recruiter_data}")
    
    print("\n--- Scraping Custom Contact Page Content ---")
    contact_data = scraper.scrape_contact_page_content(f"{base_url}/pages/contact.html")
    print(f"Contact Page Data: {contact_data}")

    # --- Using the generic methods on the index page ---
    print("\n--- Scraping Generic Content from Index Page ---")
    index_url = f"{base_url}/index.html"
    print(f"Scraping headlines from: {index_url}")
    headlines = scraper.scrape_headlines(index_url)
    print(f"Headlines: {headlines}")
    
    print(f"\nScraping links from: {index_url}")
    links = scraper.scrape_links(index_url)
    print(f"Links: {links}")
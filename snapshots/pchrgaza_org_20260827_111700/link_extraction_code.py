def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://pchrgaza.org/ar/category/genocide-on-gaza-ar/testimonies-from-the-war-ar/'

    seen = set()
    article_links = []

    # Article cards are identified by the 'news-item' class
    for card in soup.select('div.news-item'):
        # The link with class 'title' typically contains both the URL and the text
        a_tag = card.select_one('a.title')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)

        # Exclude category archives, tag/author pages, pagination, and the listing page itself
        if any(x in url for x in ['/category/', '/tag/', '/author/', '/page/']):
            continue

        if url in seen:
            continue

        # Title is primarily from the anchor text
        title = a_tag.get_text(strip=True)
        
        # Fallback: if anchor text is empty, look for title in other elements within the card
        if not title:
            # Check for common title containers within the card structural map
            title_el = card.select_one('h2, h3, p, .title')
            title = title_el.get_text(strip=True) if title_el else None

        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
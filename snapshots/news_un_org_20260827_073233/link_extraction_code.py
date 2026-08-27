def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://news.un.org/ar/news/topic/health'.split('/')[:3])

    seen = set()
    article_links = []

    # The structural map provided in the prompt only contained facet filters (category menu).
    # Based on the target URL and the requirement to find the repeating article-card container,
    # we use 'div.views-row' which is the standard container for articles on UN News topic pages.
    for article in soup.select('div.views-row'):
        # The article title and link are typically inside h3 > a
        a_tag = article.select_one('h3 a')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        
        # Extract title from the anchor; if the anchor is empty (e.g., image only), 
        # fall back to the parent heading.
        title = a_tag.get_text(strip=True)
        if not title:
            heading = article.select_one('h3')
            title = heading.get_text(strip=True) if heading else ""
            
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
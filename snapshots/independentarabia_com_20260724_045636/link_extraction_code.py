def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7'
    base_url = 'https://www.independentarabia.com'

    seen = set()
    article_links = []

    # The repeating container for articles identified in the structural map is 'article.article-item'
    for card in soup.select('article.article-item'):
        # Find the link. In the structural map, <a> tags are inside .article-item-img or the card itself.
        a_tag = card.find('a', href=True)
        if not a_tag:
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplicate and filter out the listing page itself
        if url in seen or url == target_url:
            continue
            
        # Filter out archive-like patterns or pagination
        if any(x in url for x in ['/category/', '/tag/', '/author/']) or '/page/' in url:
            continue
            
        # Title location: Prefer .article-item-title as seen in the structural map, 
        # fall back to the link's own text.
        title_el = card.select_one('.article-item-title')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
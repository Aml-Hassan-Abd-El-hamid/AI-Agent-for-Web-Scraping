def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://abduzeedo.com/'
    
    seen = set()
    article_links = []
    
    # The structural map identifies articles as 'article.post-item'
    for article in soup.select('article.post-item'):
        # The title and primary link are inside h2.post-title a
        a_tag = article.select_one('h2.post-title a')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        
        # Basic exclusions
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplicate and exclude the base listing page
        if url in seen or url == base_url:
            continue
            
        # Exclude category, tag, author archives and pagination patterns
        if any(x in url for x in ['/category/', '/tag/', '/author/', '/page/']):
            continue
            
        # Extract title from the anchor text
        title = a_tag.get_text(strip=True)
        if not title:
            # Fallback: check if there are other headings in the card
            title_el = article.select_one('h2, h3, p')
            title = title_el.get_text(strip=True) if title_el else None
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})
        
    return {"article_links": article_links}
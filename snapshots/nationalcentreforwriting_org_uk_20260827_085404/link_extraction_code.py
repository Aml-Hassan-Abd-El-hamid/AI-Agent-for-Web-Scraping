def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://nationalcentreforwriting.org.uk/writing-hub/'
    
    seen = set()
    article_links = []

    # Identify the repeating container from the structural map
    for item in soup.select('div.writing_hub_list_item'):
        # The title and primary link are located in the div with class 'title'
        a_tag = item.select_one('div.title a')
        
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        
        # Basic filters for navigation, anchors, or scripts
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplicate and exclude the listing page itself
        if url in seen or url == base_url.rstrip('/') or url == base_url:
            continue
            
        # Exclude category, tag, or author archives
        if any(x in url for x in ['/category/', '/tag/', '/author/']):
            continue
            
        # Extract title from the anchor text
        title = a_tag.get_text(strip=True)
        
        # Fallback: if a_tag had no text, try to find other text in the card (though map shows it's here)
        if not title:
            title_el = item.select_one('div.title')
            title = title_el.get_text(strip=True) if title_el else None

        if title:
            seen.add(url)
            article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
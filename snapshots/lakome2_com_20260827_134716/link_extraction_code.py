def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://lakome2.com'
    
    seen = set()
    article_links = []
    
    # Based on the structural map, article links are consistently identified by the class 'link'
    for a_tag in soup.select('a.link'):
        href = a_tag.get('href')
        if not href:
            continue
        
        # Exclude javascript and anchor links
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclude navigation, pagination, category/tag/author archives, and social media
        if any(x in url for x in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        if any(domain in url for domain in ['facebook.com', 'twitter.com', 'youtube.com', 'instagram.com']):
            continue
        if url == 'https://lakome2.com/category/art/':
            continue
        
        if url in seen:
            continue
        
        # Title extraction logic:
        # 1. Try the link's own text
        # 2. Try sibling/parent elements identified in the map: .desc, .the-text, .text
        title = a_tag.get_text(strip=True)
        if not title:
            parent = a_tag.find_parent()
            if parent:
                title_el = parent.select_one('.desc, .the-text, .text')
                if title_el:
                    title = title_el.get_text(strip=True)
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})
        
    return {"article_links": article_links}
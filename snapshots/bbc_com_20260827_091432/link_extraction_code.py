def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.bbc.com/arabic/topics/cqywj97d487t'
    base_url = target_url

    seen = set()
    article_links = []

    # The structural map shows articles are contained in li.css-psvf5b
    for card in soup.select('li.css-psvf5b'):
        # The link is inside an h2 with specific classes, and the <a> has specific classes
        a_tag = card.select_one('h2.css-1iwxh8m.ez3pb4d0 a.css-1i4ie53.eq53xv90')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        
        # Basic filters
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclusion criteria
        if url == target_url:
            continue
        if any(x in url for x in ['/category/', '/tag/', '/author/']):
            continue
        if re.search(r'/page/\d+/?$', url):
            continue
        
        if url in seen:
            continue
            
        # Extract title from the <a> tag's text
        title = a_tag.get_text(strip=True)
        
        # Fallback for title if anchor is empty (though not indicated in map for this site)
        if not title:
            title_el = card.select_one('h2, p')
            title = title_el.get_text(strip=True) if title_el else None
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
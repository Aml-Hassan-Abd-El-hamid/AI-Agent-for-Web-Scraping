def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.aajeg.com/news/palestine'.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container is the div with class 'row bs-2col node'
    for card in soup.select('div.row.bs-2col.node'):
        # The title and primary link are located within the h2 tag
        a_tag = card.select_one('h2 a')
        if not a_tag or not a_tag.get('href'):
            continue
        
        href = a_tag['href']
        
        # Filtering based on rules
        if href.startswith('javascript:') or href == '#':
            continue
        if any(pattern in href for pattern in ['/category/', '/tag/', '/author/', '/page/']):
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplicate and exclude listing page
        if url in seen or url == 'https://www.aajeg.com/news/palestine':
            continue
        
        # Title is the text of the anchor in the h2
        title = a_tag.get_text(strip=True)
        if not title:
            # Fallback: look for any other text in the card if the link was empty
            # (though in this map, h2 a has the text)
            title_el = card.select_one('p.field--name-field-summary')
            title = title_el.get_text(strip=True) if title_el else None
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
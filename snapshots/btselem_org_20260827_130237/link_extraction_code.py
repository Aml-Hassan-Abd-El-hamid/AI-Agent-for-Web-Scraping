def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.btselem.org/ota/100/all'.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container is 'div.topic-item-container'
    for card in soup.select('div.topic-item-container'):
        # The primary link and title are usually in h3.topic-iten-title a
        a_tag = card.select_one('h3.topic-iten-title a')
        
        # Fallback: if h3 a is missing, try the image link inside the card
        if not a_tag:
            a_tag = card.select_one('div.fill.image a')
            
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclusions: listing page itself, archives, and internal filters
        if url == 'https://www.btselem.org/ota/100/all' or url == 'https://www.btselem.org/ota/100/all/':
            continue
        if any(x in url for x in ['/category/', '/tag/', '/author/']):
            continue
            
        if url in seen:
            continue

        # Title Extraction: prioritize the link text, then search for a sibling header/paragraph in the card
        title = a_tag.get_text(strip=True)
        if not title:
            # Look for the title in a sibling heading/paragraph in the same card if the anchor text is empty (e.g., image link)
            title_el = card.select_one('h3.topic-iten-title, h2, p')
            title = title_el.get_text(strip=True) if title_el else ""
            
        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Base URL derived from https://akhbaralaan.net/author/wassim
    base_url = '/'.join('https://akhbaralaan.net/author/wassim'.split('/')[:3])

    seen = set()
    article_links = []

    # Each article is contained in a <li> with the class 'card'
    for card in soup.select('li.card'):
        # The link containing the title is inside the wrapper__details div
        a_tag = card.select_one('.wrapper__details a')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        
        # Exclude javascript, anchors, and archive-like patterns
        if href.startswith('javascript:') or href == '#':
            continue
        if any(pattern in href for pattern in ['/category/', '/tag/', '/author/']):
            continue
            
        url = urljoin(base_url, href)
        if url in seen:
            continue
            
        # Title is inside the h3 within the anchor
        title_el = a_tag.select_one('h3')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
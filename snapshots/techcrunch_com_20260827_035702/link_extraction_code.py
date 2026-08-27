def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://techcrunch.com/category/artificial-intelligence/'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # The structural map shows articles are contained in <li> elements with class 'wp-block-post'
    for card in soup.select('li.wp-block-post'):
        # Find the first link in the card that has an href
        a_tag = card.find('a', href=True)
        if not a_tag:
            continue
            
        href = a_tag['href']
        
        # Basic validation for junk links
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplication and Exclusion Rules
        if url in seen:
            continue
        if url == target_url.rstrip('/'):
            continue
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/']):
            continue
        if re.search(r'/page/\d+/?$', url):
            continue
            
        # TITLE location: look for h3.loop-card__title in the card first (per structural map)
        # fallback to the anchor text if the heading is missing.
        title_el = card.select_one('h3.loop-card__title')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
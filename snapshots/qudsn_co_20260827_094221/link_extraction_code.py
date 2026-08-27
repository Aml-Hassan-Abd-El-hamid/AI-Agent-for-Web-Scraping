def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://qudsn.co/post/category/6024/%D9%85%D8%AA%D8%A7%D8%A8%D8%B9%D8%A7%D8%AA-%D9%82%D8%AF%D8%B3'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    for card in soup.select('article.card'):
        # Priority: the link that contains the title text
        a_tag = card.select_one('a.card__title')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplicate and filter out navigation/archives
        if url in seen:
            continue
        if url == target_url:
            continue
        if any(x in url for x in ['/category/', '/tag/', '/author/']):
            continue
            
        # Extract title from the anchor text
        title = a_tag.get_text(strip=True)
        
        # Fallback: if anchor is empty, look for other identifying text in the card
        if not title:
            # According to map, the content is within the card, but we check standard siblings
            title_el = card.select_one('h2, h3, p, .card__title')
            title = title_el.get_text(strip=True) if title_el else ""
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
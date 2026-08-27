def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://alsifr.org/kam-kaif'.split('/')[:3])
    target_url = 'https://alsifr.org/kam-kaif'

    seen = set()
    article_links = []

    # The structural map indicates articles are within containers with class 'teaser'
    for article in soup.select('article.teaser'):
        a_tag = article.select_one('a.teaser__link')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplicate and exclude the current listing page URL
        if url in seen or url == target_url:
            continue
            
        # Rule: Title is usually in the anchor text, but check for specific title elements in the card
        title_el = article.select_one('h2.teaser__title')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.rogerebert.com/reviews'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container is article.review-small-card
    for card in soup.select('article.review-small-card'):
        a_tag = card.select_one('a.image-hover')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Filter out non-article links as per rules
        if url in seen:
            continue
        if url == target_url or url == target_url + '/':
            continue
        if any(x in url for x in ['/category/', '/tag/', '/author/']):
            continue
        
        # Title logic: look for h3 within the card (which is inside the a tag here)
        title_el = card.select_one('h3.text-2xl')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
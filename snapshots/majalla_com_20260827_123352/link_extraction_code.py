def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.majalla.com/sections/%D8%B3%D9%8A%D8%A7%D8%B3%D8%A9'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container for articles is 'article.article-item'
    for article in soup.select('article.article-item'):
        # Link is found within the image container
        a_tag = article.select_one('.article-item__img a')
        if not a_tag or not a_tag.get('href'):
            continue
        
        href = a_tag['href']
        
        # Basic filters for non-article links
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclude listing page, archive patterns, and pagination
        if url == target_url:
            continue
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        if url in seen:
            continue
            
        # Title is located in the .article-item__title div
        title_el = article.select_one('.article-item__title')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
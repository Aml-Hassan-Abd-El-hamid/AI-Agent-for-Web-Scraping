def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://www.independentarabia.com'
    target_url = 'https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7'

    seen = set()
    article_links = []

    # The structural map identifies 'article.article-item' as the repeating container
    for article in soup.select('article.article-item'):
        a_tag = article.find('a', href=True)
        if not a_tag:
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclude pagination (links containing ?page=), categories, tags, author archives, and the listing page itself
        if 'page=' in url or any(x in url for x in ['/category/', '/tag/', '/author/']):
            continue
        if url == target_url:
            continue
            
        if url in seen:
            continue
            
        # According to the map, titles are located in a div with class 'article-item-title'
        title_el = article.select_one('.article-item-title')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
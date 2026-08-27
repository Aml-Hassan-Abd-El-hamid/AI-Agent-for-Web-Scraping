def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.almasryalyoum.com/news/index?typeid=1&sectionid=10'.split('/')[:3])

    seen = set()
    article_links = []

    # The structural map shows a repeating pattern in div.all-content
    for container in soup.select('div.all-content'):
        # The title and link are located within div.article-title a
        a_tag = container.select_one('div.article-title a')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        if url in seen:
            continue
        
        # The title is inside an h2 within the anchor tag
        title_el = a_tag.select_one('h2')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
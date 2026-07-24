def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.independentarabia.com/%D8%AB%D9%82%D8%A7%D9%81%D8%A9/%D8%B3%D9%8A%D9%86%D9%85%D8%A7'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # Identify containers based on the structural map: article.article-item
    for article in soup.select('article.article-item'):
        # The link is found inside the image container
        a_tag = article.select_one('div.article-item-img a')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)

        # Exclude pagination, category/tag/author archives, and the listing page itself
        if (url == target_url or 
            '/page/' in url or 
            '/category/' in url or 
            '/tag/' in url or 
            '/author/' in url):
            continue

        if url in seen:
            continue

        # Title is typically in the sibling div.article-item-title
        title_el = article.select_one('div.article-item-title')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)

        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
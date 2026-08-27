def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://euromedmonitor.org/ar/category/26/%D8%A7%D9%84%D9%86%D8%B2%D8%A7%D8%B9%D8%A7%D8%AA-%D8%A7%D9%84%D9%85%D8%B3%D9%84%D8%AD%D8%A9'.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating article-card container on this site is .post-item
    for article in soup.select('.post-item'):
        # Target the anchor tag within the post title
        a_tag = article.select_one('.post-title a') or article.select_one('h3 a') or article.select_one('h2 a')
        
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        
        # Extract text from the anchor; fallback to sibling heading/paragraph if anchor is empty
        title = a_tag.get_text(strip=True)
        if not title:
            heading = article.select_one('h2, h3, p')
            if heading:
                title = heading.get_text(strip=True)
        
        if not title:
            continue

        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
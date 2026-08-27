def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.thearticle.com/'.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container for articles is <article class="article-listing ...">
    for article in soup.select('article.article-listing'):
        # The primary link and title are located within h4.entry-title a
        a_tag = article.select_one('h4.entry-title a')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        
        # Exclude non-article links based on criteria
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclude category, tag, author archive, sponsors, and the root URL
        exclude_patterns = ['/exchange/', '/contributor/', '/sponsors', '/category/', '/tag/', '/author/']
        if any(pattern in url for pattern in exclude_patterns) or url == base_url or url == base_url + '/':
            continue
            
        if url in seen:
            continue
        
        # Get title from the link text
        title = a_tag.get_text(strip=True)
        
        # Fallback: if the link had no text (though in this map it does), 
        # we would look for sibling headings or the data-href/id attributes.
        if not title:
            title_el = article.select_one('h4.entry-title')
            title = title_el.get_text(strip=True) if title_el else None
            
        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://engineering.indeedblog.com/blog/category/engineering/'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # Repeating container for articles is <article class="clearfix">
    for article in soup.select('article.clearfix'):
        # Title and link are within <h2 class="entry-title"> <a>
        a_tag = article.select_one('h2.entry-title a')
        if not a_tag or not a_tag.get('href'):
            continue
        
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplication and Exclusions
        if url in seen:
            continue
        if url == target_url:
            continue
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        
        # title is the link's own text
        title = a_tag.get_text(strip=True)
        
        # Fallback if anchor text is empty (look for title in the card)
        if not title:
            title_el = article.select_one('h2.entry-title, h3, p.the-title')
            title = title_el.get_text(strip=True) if title_el else None
            
        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
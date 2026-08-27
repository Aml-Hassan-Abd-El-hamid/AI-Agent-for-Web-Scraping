def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://arabic.euronews.com/culture'.split('/')[:3])

    seen = set()
    article_links = []

    # Containers are article tags with class 'the-media-object'
    for article in soup.select('article.the-media-object'):
        # Prioritize the link that typically contains the title
        a_tag = article.select_one('a.the-media-object__link')
        
        # Fallback to the image link in the figure if the title link is missing
        if not a_tag:
            a_tag = article.select_one('figure a')

        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        
        # Exclude navigation, category/tag/author archives, and structural markers
        if any(pattern in href for pattern in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        if href.startswith('javascript:') or href == '#':
            continue
        
        url = urljoin(base_url, href)
        
        # Exclude the listing page's own URL
        if url == 'https://arabic.euronews.com/culture':
            continue
            
        if url in seen:
            continue

        # Title extraction logic:
        # 1. Try the anchor's own text
        # 2. If anchor has no text (e.g. image wrap), look for sibling/container text in the card
        title = a_tag.get_text(strip=True)
        if not title:
            content_div = article.select_one('div.the-media-object__content')
            if content_div:
                title = content_div.get_text(strip=True)
        
        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
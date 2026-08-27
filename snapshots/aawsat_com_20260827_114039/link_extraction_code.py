def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://aawsat.com/%D8%A7%D9%84%D8%B1%D8%A3%D9%8A'.split('/')[:3])

    seen = set()
    article_links = []

    # The structural map indicates that article containers have the class 'node--type-opinion'
    # and contain the URL in the 'data-vr-contentbox-url' attribute.
    articles = soup.select('div.node--type-opinion')

    for article in articles:
        # 1. Extract URL
        url = article.get('data-vr-contentbox-url')
        if not url:
            # Fallback to find any anchor tag within the card
            a_tag = article.find('a', href=True)
            if a_tag:
                url = urljoin(base_url, a_tag['href'])
        
        if not url or url.startswith('javascript:') or url == '#':
            continue
            
        if url in seen:
            continue
        seen.add(url)

        # 2. Extract Title
        # Prioritize: h2 in title div -> title div text -> info div text
        title_tag = article.select_one('.article-item-title h2') or \
                    article.select_one('.article-item-title') or \
                    article.select_one('.article-item-info')
        
        if not title_tag:
            continue
            
        title = title_tag.get_text(strip=True)
        
        # The .article-item-info often contains extra text like "استمع إلى المقالة"
        # We clean that up to get the actual title.
        if "استمع إلى المقالة" in title:
            title = title.split("استمع إلى المقالة")[0].strip()

        if not title:
            continue

        article_links.append({"url": url, "title": title})

    # Cap at 500 to strictly adhere to output schema limits mentioned in the error
    return {"article_links": article_links[:500]}
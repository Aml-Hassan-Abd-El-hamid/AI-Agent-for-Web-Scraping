def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://lithub.com'

    seen = set()
    article_links = []

    # The structural map shows article cards with these specific classes
    containers = soup.select('.featured-story, .secondary-story, .tertiary-story, .featured-item-story')

    for card in containers:
        # The title and link are consistently located in an h2 tag within these containers
        a_tag = card.select_one('h2 a')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)

        # Filter out pagination, authors, categories, and external/non-article domains
        if url in seen:
            continue
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        if url == 'https://lithub.com/category/fictionandpoetry/short-story/':
            continue
        if 'lithub.com' not in url:
            continue

        seen.add(url)
        title = a_tag.get_text(strip=True)
        
        # Fallback: if h2 a had no text, check for other heading/p within the same card
        if not title:
            title_el = card.select_one('h3, p')
            title = title_el.get_text(strip=True) if title_el else None
        
        if title:
            article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
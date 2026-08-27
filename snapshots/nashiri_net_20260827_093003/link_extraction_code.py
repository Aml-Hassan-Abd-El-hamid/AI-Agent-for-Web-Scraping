def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.nashiri.net/index.php/articles/literature-and-art'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container for articles is div.item.column-1
    for card in soup.select('div.item.column-1'):
        # The primary article link and title are within div.page-header h2 a
        a_tag = card.select_one('div.page-header h2 a')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)

        # Exclusion rules
        # 1. Exclude the listing page's own URL
        if url == target_url or url == target_url + '/':
            continue
        # 2. Exclude category/tag/author archives
        if any(token in url for token in ['/category/', '/tag/', '/author/']):
            continue
        # 3. Deduplicate
        if url in seen:
            continue

        # Title location: attempt to get text from the anchor, then fallback to card headings
        title = a_tag.get_text(strip=True)
        if not title:
            title_el = card.select_one('h2, h3, p')
            title = title_el.get_text(strip=True) if title_el else None

        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
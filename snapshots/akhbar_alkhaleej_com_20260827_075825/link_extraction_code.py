def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://akhbar-alkhaleej.com/news/section/BUSI'

    seen = set()
    article_links = []

    # Repeating container identified as div.bah-news-box
    for card in soup.select('div.bah-news-box'):
        # The anchor is present in both the image container and the h3 title
        a_tag = card.select_one('h3 a, .imagecontainer a')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(target_url, href)

        # Filter out navigation, pagination, archives, and the listing page itself
        if url == target_url:
            continue
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        # Check for pagination numeric text (handled via URL pattern above, but as safety)
        
        if url in seen:
            continue

        # Title location: Prefer text from the h3 anchor if available
        title_el = card.select_one('h3 a')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)

        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
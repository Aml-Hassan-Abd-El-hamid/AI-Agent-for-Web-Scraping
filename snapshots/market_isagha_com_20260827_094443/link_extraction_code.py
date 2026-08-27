def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://market.isagha.com/articles'
    base_url = target_url

    seen = set()
    article_links = []

    # The repeating container is article.blog-post
    for article in soup.select('article.blog-post'):
        # The title and primary link are located in div.post-heading h2 a
        link_tag = article.select_one('div.post-heading h2 a')
        if not link_tag:
            continue

        href = link_tag.get('href')
        if not href:
            continue

        # Exclude javascript and anchor-only links
        if href.startswith('javascript:') or href == '#':
            continue

        # Exclude category, tag, author, and pagination links
        # Based on rules: /category/, /tag/, /author/, and pagination patterns
        if any(keyword in href for keyword in ['/category/', '/tag/', '/author/', '?author=', '?page=']):
            continue

        url = urljoin(base_url, href)

        # Exclude the listing page's own URL
        if url == target_url or url == target_url + '/':
            continue

        if url in seen:
            continue

        # Extract title from the anchor text
        title = link_tag.get_text(strip=True)
        
        # If anchor text is empty, look for title in sibling elements within the card
        if not title:
            title_el = article.select_one('h2, h3, p')
            title = title_el.get_text(strip=True) if title_el else ""

        if title:
            seen.add(url)
            article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
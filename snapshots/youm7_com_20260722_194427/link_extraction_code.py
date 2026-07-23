def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # Identify containers: main articles in div.bigOneSec and ticker articles in ul#nt-title li
    # Both selectors appear in the structural map.
    containers = soup.select('div.bigOneSec, ul#nt-title li')

    for container in containers:
        a_tag = container.find('a', href=True)
        if not a_tag:
            continue

        href = a_tag['href']
        url = urljoin(base_url, href)

        # Filter: Deduplicate
        if url in seen:
            continue
        
        # Filter: Javascript/Anchor links
        if href.startswith('javascript:') or href == '#':
            continue

        # Filter: Navigation, Pagination, Categories, and own URL
        # youm7 category/archive URLs typically contain '/Section/'
        if '/Section/' in url:
            continue
        if '/page/' in url:
            continue
        if url == target_url:
            continue

        # Title: Prefer the container's text (which contains the headline in the map) 
        # or fall back to the anchor's text.
        title = container.get_text(strip=True)
        if not title:
            title = a_tag.get_text(strip=True)

        if title:
            seen.add(url)
            article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
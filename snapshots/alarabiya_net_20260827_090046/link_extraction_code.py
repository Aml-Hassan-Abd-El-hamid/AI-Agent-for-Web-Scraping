def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.alarabiya.net/views'
    base_url = 'https://www.alarabiya.net'

    seen = set()
    article_links = []

    # The structural map indicates that article links are <a> tags with class 'opinion_title'
    # inside containers with class 'opinion_link'.
    for container in soup.select('div.opinion_link'):
        a_tag = container.select_one('a.opinion_title')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        
        # Exclude javascript, anchor-only links, and archive links (category, tag, author)
        if href.startswith('javascript:') or href == '#':
            continue
        if any(pattern in href for pattern in ['/category/', '/tag/', '/author/']):
            continue

        url = urljoin(base_url, href)
        
        # Exclude the listing page's own URL and duplicates
        if url == target_url or url in seen:
            continue

        # Extract title from the link text. 
        # If empty, we would look for a sibling title, but in this map a.opinion_title contains the text.
        title = a_tag.get_text(strip=True)
        if not title:
            # Fallback to other text elements in the card if anchor text is missing
            title_el = container.select_one('p, span, h2, h3')
            title = title_el.get_text(strip=True) if title_el else ""

        if title:
            seen.add(url)
            article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
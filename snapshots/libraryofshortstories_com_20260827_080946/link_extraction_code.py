def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.libraryofshortstories.com/stories'.split('/')[:3])

    seen = set()
    article_links = []

    # Each story is contained within a div.story_tile_cluster
    for cluster in soup.select('div.story_tile_cluster'):
        # The primary link to the story reader
        a_tag = cluster.select_one('a.story_tile.true_link')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        # Basic filters for invalid or non-article links
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)
        
        # Deduplicate and exclude navigation/archive patterns
        if url in seen:
            continue
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        if url == 'https://www.libraryofshortstories.com/stories':
            continue

        # Title is located within div.story_tile_title inside the anchor
        title_el = a_tag.select_one('div.story_tile_title')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            # Fallback to anchor text if the specific title div isn't found
            title = a_tag.get_text(strip=True)

        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
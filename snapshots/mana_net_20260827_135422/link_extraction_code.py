def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://mana.net/category/articles/'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # All articles are contained within <article class="jeg_post">
    for card in soup.select('article.jeg_post'):
        # Primary source for title and URL is the link inside the post title heading
        title_link = card.select_one('h2.jeg_post_title a, h3.jeg_post_title a')
        
        if title_link:
            href = title_link.get('href')
            title = title_link.get_text(strip=True)
        else:
            # Fallback: take the first available link and look for a title in siblings (h2, h3, p)
            a_tag = card.find('a', href=True)
            if not a_tag:
                continue
            href = a_tag.get('href')
            # Look for title in the card structure as per the rule for image-card layouts
            title_el = card.select_one('h2, h3, p')
            title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)

        if not href or not title:
            continue

        # Filter out non-article links
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)

        # Deduplicate and apply strict exclusions
        if url in seen:
            continue
        if url == target_url:
            continue
        # Exclude category/tag/author archives and pagination links
        if any(p in url for p in ['/category/', '/tag/', '/author/', '/page/']):
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
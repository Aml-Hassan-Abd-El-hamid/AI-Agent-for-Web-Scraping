def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Derive base_url from target URL
    target_url = 'https://alqaheranews.net/category/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # Identify repeating container based on structural map: <article>
    for article in soup.find_all('article'):
        # The link is usually in a.img-link
        a_tag = article.select_one('a.img-link')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']

        # Exclusions: javascript, anchors, categories, tags, authors, pagination
        if (href.startswith('javascript:') or 
            href == '#' or 
            '/category/' in href or 
            '/tag/' in href or 
            '/author/' in href or 
            re.search(r'/page/\d+/?$', href)):
            continue

        url = urljoin(base_url, href)
        
        # Exclude the listing page's own URL and duplicates
        if url == target_url or url in seen:
            continue

        # TITLE location: anchor text or sibling container .post-card-content
        title_el = article.select_one('.post-card-content')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)

        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
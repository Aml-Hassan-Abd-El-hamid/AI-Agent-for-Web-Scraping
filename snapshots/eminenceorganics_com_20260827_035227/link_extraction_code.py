def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://eminenceorganics.com/us/blog/celebrity-skincare'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container for articles is the <li> inside the <ul> with class 'b-blog-grid'
    for item in soup.select('ul.b-blog-grid li.l-col-12'):
        a_tag = item.find('a', href=True)
        if not a_tag:
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)

        # Exclude duplicates, the listing page itself, and non-article links
        if url in seen:
            continue
        if url == target_url:
            continue
        if any(pattern in url for pattern in ['/category/', '/tag/', '/author/']):
            continue
        if re.search(r'/page/\d+/?$', url):
            continue

        # Title: check for common heading/paragraph tags in the card first, fallback to anchor text
        title_el = item.select_one('h2, h3, p, .title')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)

        if not title:
            continue

        # Final check for pagination text (e.g., "1", "2")
        if title.isdigit():
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
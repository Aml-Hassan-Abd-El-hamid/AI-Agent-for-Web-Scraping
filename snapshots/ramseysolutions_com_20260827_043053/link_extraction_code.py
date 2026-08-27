def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.ramseysolutions.com/articles'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    for card in soup.select('article.rds-Card_1'):
        # Attempt to find the primary link and title in the h3
        a_tag = card.select_one('h3.rds-Card-title_1 a')
        
        # Fallback to image wrapper if h3 link is missing
        if not a_tag:
            a_tag = card.select_one('a.rds-Card-imageWrapper_1')
            
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue

        url = urljoin(base_url, href)

        # Exclusions: deduplication, target URL, archives, pagination
        if url == target_url or url in seen:
            continue
        if any(x in url for x in ['/category/', '/tag/', '/author/']) or re.search(r'/page/\d+/?$', url):
            continue

        # Title extraction: Priority to the h3 element's text
        title_el = card.select_one('h3.rds-Card-title_1')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
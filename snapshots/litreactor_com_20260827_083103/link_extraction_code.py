def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://litreactor.com/columns/'.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container is the div with class article-entry
    for card in soup.select('div.article-entry'):
        # The primary link contains the title and descriptive text
        a_tag = card.select_one('a.no-decoration')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclude pagination, categories, tags, authors, and the listing page itself
        if (url in seen or 
            '/page/' in url or 
            '/category/' in url or 
            '/tag/' in url or 
            '/author/' in url or 
            url == 'https://litreactor.com/columns/'):
            continue

        # The title is located within the h3 tag inside the anchor
        title_el = a_tag.select_one('h3.mb-sm')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)

        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://mawdoo3.com/%D8%AA%D8%B5%D9%86%D9%8A%D9%81:%D8%A7%D9%84%D8%A2%D8%AF%D8%A7%D8%A8'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container is li.columns.large-4.medium-3.small-6
    for li in soup.select('li.columns.large-4.medium-3.small-6'):
        a_tag = li.select_one('a.category-box')
        if not a_tag or not a_tag.get('href'):
            continue
        
        href = a_tag['href']
        # Exclude navigation, JS, and anchor links
        if href.startswith('javascript:') or href == '#':
            continue
        
        # Exclude category, tag, and author archive links
        if any(pattern in href for pattern in ['/category/', '/tag/', '/author/']):
            continue
            
        url = urljoin(base_url, href)
        # Deduplicate and exclude the listing page's own URL
        if url in seen or url == target_url:
            continue
        
        # Title is located inside div.title within the anchor
        title_el = a_tag.select_one('div.title')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})
        
    return {"article_links": article_links}
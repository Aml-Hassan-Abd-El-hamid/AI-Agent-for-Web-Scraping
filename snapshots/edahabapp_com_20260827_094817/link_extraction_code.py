def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://edahabapp.com/articles'

    seen = set()
    article_links = []

    # The repeating container is a div with these specific utility classes
    containers = soup.select('div.bg-navy-light.rounded-lg.overflow-hidden.shadow')

    for card in containers:
        # The article link with the title is inside the div.p-4
        a_tag = card.select_one('div.p-4 a')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        
        # Filter out unwanted links
        if (href.startswith('javascript:') or 
            href == '#' or 
            any(x in href for x in ['/category/', '/tag/', '/author/', '/page/'])):
            continue
            
        url = urljoin(base_url, href)
        
        # Avoid duplicates and the listing page itself
        if url in seen or url == base_url.rstrip('/'):
            continue
            
        # The title is the text of the anchor tag in the p-4 div
        title = a_tag.get_text(strip=True)
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
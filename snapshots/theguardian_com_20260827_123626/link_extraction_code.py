def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://www.theguardian.com'

    seen = set()
    article_links = []

    # Each article item is contained within a div with class 'dcr-7fugik' 
    # which houses both the anchor link and the title.
    for container in soup.select('div.dcr-7fugik'):
        a_tag = container.select_one('a.dcr-idxb0f')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        
        # Exclusion filters
        if href.startswith('javascript:') or href == '#':
            continue
        if any(x in href for x in ['/category/', '/tag/', '/author/']):
            continue
        if re.search(r'/page/\d+/?$', href):
            continue
            
        url = urljoin(base_url, href)
        if url == 'https://www.theguardian.com/world/gaza' or url in seen:
            continue
            
        # Title extraction: Look for span.headline-text first, then h3.card-headline
        title_el = container.select_one('span.headline-text, h3.card-headline')
        if title_el:
            title = title_el.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
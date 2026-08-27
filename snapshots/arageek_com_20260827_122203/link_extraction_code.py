def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://www.arageek.com'
    
    seen = set()
    article_links = []
    
    # The structural map identifies articles with the class 'chakra-linkbox css-ar-10n1vic'
    # Each of these contains an <a> tag with class 'chakra-linkbox__overlay css-ar-1hnz6hu'
    for article in soup.select('article.css-ar-10n1vic'):
        a_tag = article.select_one('a.css-ar-1hnz6hu')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        
        # Filter out navigation, anchors, and javascript
        if href.startswith('javascript:') or href == '#' or any(x in href for x in ['/category/', '/tag/', '/author/']):
            continue
            
        url = urljoin(base_url, href)
        if url in seen:
            continue
            
        # Title location: first try the anchor text, then the heading span in the card
        title = a_tag.get_text(strip=True)
        if not title:
            title_el = article.select_one('span.css-ar-knslkk')
            title = title_el.get_text(strip=True) if title_el else ""
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})
        
    return {"article_links": article_links}
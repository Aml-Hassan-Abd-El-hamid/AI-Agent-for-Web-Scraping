def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = 'https://acpss.ahram.org.eg/OuterWriter/28/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA/0.aspx'

    seen = set()
    article_links = []

    # Each article entry is contained within a div with class 'col-6 d-flex'
    for card in soup.select('div.col-6.d-flex'):
        # The article link is located within the 'col-lg-9 align-self-center' div
        content_div = card.select_one('div.col-lg-9.align-self-center')
        if not content_div:
            continue
        
        # Find all links within the content div
        links = content_div.find_all('a', href=True)
        for a in links:
            href = a['href']
            
            # Filter out author/archive links (containing '/Writer/') 
            # and common navigation patterns
            if '/Writer/' in href:
                continue
            if '/category/' in href or '/tag/' in href or '/author/' in href:
                continue
            if href.startswith('javascript:') or href == '#':
                continue
                
            # Ensure we are targeting news articles (containing '/News/')
            if '/News/' not in href:
                continue
                
            url = urljoin(base_url, href)
            if url in seen:
                continue
                
            # The title is within the <a> tag (specifically inside an <h4> in the map)
            title = a.get_text(strip=True)
            
            # If <a> had no text, we'd look for a sibling heading, 
            # but here the <h4> is a child of the <a>.
            if not title:
                h4 = a.find('h4')
                title = h4.get_text(strip=True) if h4 else ""

            if title:
                seen.add(url)
                article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
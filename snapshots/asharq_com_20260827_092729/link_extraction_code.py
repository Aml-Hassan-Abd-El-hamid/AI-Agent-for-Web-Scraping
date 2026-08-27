def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://asharq.com/politics/'.split('/')[:3])

    seen = set()
    article_links = []

    # The repeating container is the div with data-card="list-card"
    for card in soup.select('div[data-card="list-card"]'):
        # The structural map shows the link wrapping the text content is a.flex.flex-col
        a_tag = card.select_one('a.flex.flex-col')
        if not a_tag or not a_tag.get('href'):
            # Fallback to any anchor in the card if the specific class isn't found
            a_tag = card.find('a', href=True)
            if not a_tag:
                continue

        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclude navigation, category, tag, author, pagination, and the listing page itself
        if any(x in url for x in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        if url == 'https://asharq.com/politics/' or url == 'https://asharq.com/politics':
            continue
        if url in seen:
            continue

        # The title is located in the h2 tag within the card
        title_el = card.select_one('h2')
        title = title_el.get_text(strip=True) if title_el else a_tag.get_text(strip=True)
        
        if not title:
            continue

        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
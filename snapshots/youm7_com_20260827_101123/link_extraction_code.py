def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1'
    base_url = '/'.join(target_url.split('/')[:3])

    seen = set()
    article_links = []

    # The structural map shows the main container for each article is a div with class 'col-xs-12 bigOneSec'
    for card in soup.select('div.bigOneSec'):
        # The title and link are located within the h3 tag inside the card
        a_tag = card.select_one('h3 a')
        if not a_tag or not a_tag.get('href'):
            continue

        href = a_tag['href']

        # Exclude navigation, pagination, categories, tags, authors, and invalid links
        if (href.startswith('javascript:') or 
            href == '#' or 
            '/category/' in href or 
            '/tag/' in href or 
            '/author/' in href or 
            '/page/' in href):
            continue

        url = urljoin(base_url, href)

        # Exclude the listing page's own URL
        if url == target_url:
            continue

        if url in seen:
            continue

        seen.add(url)
        
        # Get the title from the anchor text
        title = a_tag.get_text(strip=True)
        
        # Fallback: if <a> has no text, look for other text in the card (though structural map shows it's in the <a>)
        if not title:
            title_el = card.select_one('h3, p')
            title = title_el.get_text(strip=True) if title_el else None

        if title:
            article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
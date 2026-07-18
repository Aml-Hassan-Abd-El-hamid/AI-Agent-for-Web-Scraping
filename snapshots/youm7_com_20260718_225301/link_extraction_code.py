def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.youm7.com/Section/%D8%A3%D8%AE%D8%A8%D8%A7%D8%B1-%D8%B9%D8%A7%D8%AC%D9%84%D8%A9/65/1'
    base_url = 'https://www.youm7.com'

    seen = set()
    article_links = []

    # The structural map identifies 'div.bigOneSec' as the article containers within 'div#paging'
    for container in soup.select('div.bigOneSec'):
        a_tag = container.find('a')
        if not a_tag or not a_tag.get('href'):
            continue
        
        href = a_tag['href']
        
        # Exclude javascript and anchor links
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Deduplicate by URL
        if url in seen:
            continue
            
        # Exclude listing page's own URL
        if url == target_url:
            continue
            
        # Exclude category/tag/author archive links (URLs containing /Section/, /category/, /tag/, /author/)
        if any(x in url for x in ['/Section/', '/category/', '/tag/', '/author/']):
            continue
            
        # Exclude social media domains
        social_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'youtube.com', 'itunes.apple.com', 'play.google.com', 'appgallery.cloud.huawei.com']
        if any(domain in url for domain in social_domains):
            continue

        # Exclude pagination links (urls ending in /page/N/ or similar patterns)
        # based on the rule "URL ends in /page/N/" or visible text is a number.
        # We also check the text of the link.
        link_text = a_tag.get_text(strip=True)
        if link_text.isdigit():
            continue

        seen.add(url)
        title = link_text if link_text else container.get_text(strip=True)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
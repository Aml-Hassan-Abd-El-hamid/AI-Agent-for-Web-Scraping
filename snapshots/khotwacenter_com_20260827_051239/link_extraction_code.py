def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    target_url = 'https://www.khotwacenter.com/category/%D8%A7%D9%84%D8%AF%D8%B1%D8%A7%D8%B3%D8%A7%D8%AA-%D9%88%D8%A7%D9%84%D8%A3%D8%A8%D8%AD%D8%A7%D8%AB/%D8%A8%D8%AD%D9%88%D8%AB-%D9%88%D8%AF%D8%B1%D8%A7%D8%B3%D8%A7%D8%AA/%D9%85%D9%82%D8%A7%D9%84%D8%A7%D8%AA-%D8%AB%D9%82%D8%A7%D9%81%D9%8A%D8%A9/'
    base_url = '/'.join(target_url.split('/')[:3]) + '/'

    seen = set()
    article_links = []

    # Repeating container: article.item-list
    for card in soup.select('article.item-list'):
        # Primary link and title source
        a_tag = card.select_one('h2.post-box-title a')
        if not a_tag or not a_tag.get('href'):
            continue
        
        href = a_tag['href']
        
        # Filtering basic invalid hrefs
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        
        # Exclusions: 
        # 1. Category, Tag, Author archives
        # 2. Pagination (/page/N/)
        # 3. Listing page's own URL
        # 4. Social media (though not present in structural map)
        if any(x in url for x in ['/category/', '/tag/', '/author/', '/page/']):
            continue
        if url == target_url:
            continue
            
        if url in seen:
            continue
        
        # Title extraction logic
        title = a_tag.get_text(strip=True)
        if not title:
            # Look for title in sibling headings or paragraphs within the card
            title_el = card.select_one('h2, h3, p')
            title = title_el.get_text(strip=True) if title_el else ""
            
        if not title:
            continue
            
        seen.add(url)
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
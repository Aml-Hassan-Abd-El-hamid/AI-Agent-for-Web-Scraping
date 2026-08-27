def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.masrawy.com/today#Nav-Today'.split('/')[:3])

    seen = set()
    article_links = []

    # The structural map indicates articles are in <li> tags with classes starting with "mix"
    for li in soup.select('li.mix'):
        a_tag = li.select_one('a.imageCntnr')
        if not a_tag or not a_tag.get('href'):
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)
        
        # Per instructions: if <a> has no text, take title from sibling (div.desc in this case)
        desc_tag = li.select_one('div.desc')
        if desc_tag:
            title = desc_tag.get_text(strip=True)
        else:
            title = a_tag.get_text(strip=True)
            
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
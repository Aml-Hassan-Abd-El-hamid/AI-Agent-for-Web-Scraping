def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.akhbaar24.com/%D8%AF%D9%88%D9%84%D9%8A%D8%A7%D8%AA'.split('/')[:3])

    seen = set()
    article_links = []

    # The structural map shows articles are contained in <article> tags with a specific set of classes
    # We use 'article.group' as a reliable selector for these card containers
    for article in soup.select('article.group'):
        # Find the link - both the image <a> and the title <a> point to the same article
        a_tag = article.select_one('a[href]')
        if not a_tag:
            continue
            
        href = a_tag['href']
        if href.startswith('javascript:') or href == '#':
            continue
            
        url = urljoin(base_url, href)
        if url in seen:
            continue
        seen.add(url)

        # Title extraction: Based on structural map, the title is inside an <h3> tag
        # If <h3> is missing or empty, we search for the first <a> with non-empty text
        title_tag = article.select_one('h3')
        if title_tag:
            title = title_tag.get_text(strip=True)
        else:
            title = ""
            for link in article.select('a'):
                text = link.get_text(strip=True)
                if text:
                    title = text
                    break
        
        article_links.append({"url": url, "title": title})

    return {"article_links": article_links}
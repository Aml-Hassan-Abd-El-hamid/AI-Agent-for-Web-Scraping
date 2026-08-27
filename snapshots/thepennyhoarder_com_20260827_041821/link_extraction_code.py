def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    base_url = '/'.join('https://www.thepennyhoarder.com/retirement/'.split('/')[:3])
    
    article_links = []
    seen = set()

    # 1. Extract Featured Article (Photo Essay)
    # Selector based on structural map: div.photo-essay-block -> a.photo-essay-article-content-title
    featured_blocks = soup.select('div.photo-essay-block')
    for block in featured_blocks:
        a_tag = block.select_one('a.photo-essay-article-content-title')
        if not a_tag:
            continue
        
        href = a_tag.get('href')
        if not href:
            continue
            
        url = urljoin(base_url, href)
        if url not in seen:
            title = a_tag.get_text(strip=True)
            article_links.append({"url": url, "title": title})
            seen.add(url)

    # 2. Extract Subcategory Articles
    # Selector based on structural map: div.category-subcategory-post
    for post in soup.select('div.category-subcategory-post'):
        # Title is found in the content div sibling/child of the image
        content_div = post.select_one('.category-subcategory-post-content')
        title = content_div.get_text(strip=True) if content_div else ""
        
        # The structural map doesn't explicitly list the <a> tag inside category-subcategory-post,
        # but based on the "TITLE location" rule, we must find the associated link.
        # It is typically either the post itself, a parent, or a child <a>.
        a_tag = post.find_parent('a') or post.find('a')
        
        if a_tag and a_tag.get('href'):
            href = a_tag['href']
            url = urljoin(base_url, href)
            if url not in seen:
                # Use the title from the content div if the link text is empty/missing
                final_title = title if title else a_tag.get_text(strip=True)
                article_links.append({"url": url, "title": final_title})
                seen.add(url)

    return {"article_links": article_links}
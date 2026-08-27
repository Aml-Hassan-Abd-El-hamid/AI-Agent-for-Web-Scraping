def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Derive base_url from TARGET URL: https://arabic.rt.com/russia/
    base_url = '/'.join('https://arabic.rt.com/russia/'.split('/')[:3])

    seen = set()
    article_links = []

    # Identify containers for articles from the structural map
    # 1. special-widget__article
    # 2. main-article (which also has class main-news__main-article)
    containers = soup.select('article.special-widget__article, article.main-article')

    for card in containers:
        # URL Extraction
        url = None
        # Try finding an <a> tag first
        a_tag = card.find('a', href=True)
        if a_tag:
            url = a_tag['href']
        # Fallback: use the 'data-link' attribute from the share-block as seen in the map
        elif card.select_one('div.share-block'):
            url = card.select_one('div.share-block').get('data-link')

        if not url:
            continue

        # Resolve URL
        full_url = urljoin(base_url, url)

        # Filtering Logic
        if full_url in seen:
            continue
        
        # Exclude javascript and anchor-only links
        if url.startswith('javascript:') or url == '#':
            continue
        
        # Exclude category, tag, author archives
        if any(keyword in full_url for keyword in ['/category/', '/tag/', '/author/']):
            continue
        
        # Exclude pagination (URL ending in /page/N)
        if re.search(r'/page/\d+/?$', full_url):
            continue

        # Exclude social media domains
        if any(domain in full_url.lower() for domain in ['facebook.com', 'twitter.com', 'x.com', 'vk.com', 'telegram.org']):
            continue

        # Exclude the listing page's own URL
        if full_url.rstrip('/') == 'https://arabic.rt.com/russia'.rstrip('/'):
            continue

        # Title Extraction
        # Use selectors that appear in the structural map
        title_el = card.select_one('h3.special-widget__article-title, h3.main-article__title')
        if title_el:
            title = title_el.get_text(strip=True)
        elif a_tag:
            title = a_tag.get_text(strip=True)
        else:
            title = None

        if not title:
            continue

        # Final validation and addition
        seen.add(full_url)
        article_links.append({"url": full_url, "title": title})

    return {"article_links": article_links}
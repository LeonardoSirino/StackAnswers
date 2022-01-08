import scrapy


class NorgrenSpider(scrapy.Spider):
    name = 'norgren'
    start_urls = ['https://www.norgren.com/de/en/list']

    def start_requests(self):
        for start_url in self.start_urls:
            yield scrapy.Request(start_url, callback=self.parse)

    def parse(self, response):
        link_list = []

        more_info_items = response.css(
            ".match-height a.more-info::attr(href)").getall()

        detail_items = [item for item in more_info_items if '/detail/' in item]
        if len(detail_items) > 0:
            print(f'This is a link you are searching for: {response.url}')

        for item in more_info_items:
            if not "/detail/" in item:
                inner_page_link = response.urljoin(item)
                link_list.append(inner_page_link)
                yield {"target_url": inner_page_link}

        for new_link in link_list:
            yield scrapy.Request(new_link, callback=self.parse)

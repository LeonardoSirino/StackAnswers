import scrapy


class NorgrenSpider(scrapy.Spider):
    name = 'norgren'
    start_urls = ['https://www.norgren.com/de/en/list']
    custom_settings = {
        'LOG_FILE': 'norgren.log',
        'LOG_ENABLE': False,
    }

    def start_requests(self):
        for start_url in self.start_urls:
            yield scrapy.Request(start_url, callback=self.parse)

    def parse(self, response):
        link_list = []
        for item in response.css(".match-height a.more-info::attr(href)").getall():
            if not "/detail/" in item:
                inner_page_link = response.urljoin(item)
                print(f"Found link: {inner_page_link}")
                link_list.append(inner_page_link)
                yield {"target_url": inner_page_link}

        for new_link in link_list:
            yield scrapy.Request(new_link, callback=self.parse)

